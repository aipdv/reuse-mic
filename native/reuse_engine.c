// Native streaming engine for NVIDIA Real-time RE-USE (look-ahead 0). Apple Accelerate + NEON, no PyTorch.
// One call = one STFT frame: log1p-magnitude + phase in [nb] -> model magnitude (log domain) + phase [nb].
// Mirrors reuse-mic/reuse_core.py LiveEnhancer -> StreamingSEMamba (NVIDIA online_inference.py).
//
// Layout: every activation is row-major [rows = frequency][cols = channels].
//  - 3x3 convs = one sgemm per tap on a row-shifted pointer into a zero-padded buffer (no im2col).
//  - Dense blocks keep one ring of [W+2][256] frames; channel concat is a column range, no copies:
//      cols 0..63 = y2, 64..127 = y1, 128..191 = y0, 192..255 = block input x. Conv i reads cols 192-64i..255.
//  - SPConvTranspose (r=2): conv output [Fm][128] read as [2Fm][64] is already the interleaved result.
//  - Mamba scan: one fused NEON pass per channel (16 states in 4 registers, inline exp).
// Build: see build.sh (-DNOPROF for the release library).
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>
#include <mach/mach_time.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#include <sys/mman.h>
#include <os/workgroup.h>
#include <stdatomic.h>
// All engine memory is pinned (mlock): on a busy 8 GB Mac, compressed or swapped pages
// would stall a hop for tens of ms regardless of thread priority.
static void *pin(void *p, size_t n) { if (p && n) mlock(p, n); return p; }
#define MALLOC(n) pin(malloc(n), (size_t)(n))
#define CALLOC(n, sz) pin(calloc((n), (sz)), (size_t)(n) * (size_t)(sz))

static double g_prof[8];
#ifdef NOPROF
#define PT0()
#define PT1(i)
#else
static double tb_ms(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return (double)tb.numer / tb.denom / 1e6; }
#define PT0() uint64_t _pt = mach_absolute_time()
#define PT1(i) (g_prof[i] += (mach_absolute_time() - _pt) * tb_ms())
#endif
// 0 ssm scan (incl. softplus/gate pre-pass), 1 clamped scan groups, 2 sgemm, 3 scan groups, 5 mamba conv+silu, 6 layernorm/prelu
void re_prof(double *out) { memcpy(out, g_prof, sizeof g_prof); memset(g_prof, 0, sizeof g_prof); }

#define DI 256   // mamba d_inner
#define NS 16    // d_state
#define NDBL 36  // dt_rank 4 + 2 * NS
#define RING 17  // dense ring: 2 * max dilation + 1 frames

// ---------- weights ----------
typedef struct { char *name; int ndim, dims[4]; float *data; } Tensor;
static Tensor *g_t; static int g_nt;

static int load_weights(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    char magic[4]; unsigned n;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "RUSE", 4) || fread(&n, 4, 1, f) != 1) { fclose(f); return -2; }
    g_t = CALLOC(n, sizeof(Tensor)); g_nt = (int)n;
    for (unsigned i = 0; i < n; i++) {
        unsigned len, nd, d;
        if (fread(&len, 4, 1, f) != 1) { fclose(f); return -3; }
        g_t[i].name = CALLOC(len + 1, 1); fread(g_t[i].name, 1, len, f);
        fread(&nd, 4, 1, f); g_t[i].ndim = (int)nd; long sz = 1;
        for (unsigned k = 0; k < nd; k++) { fread(&d, 4, 1, f); g_t[i].dims[k] = (int)d; sz *= d; }
        g_t[i].data = MALLOC(sz * sizeof(float));
        if ((long)fread(g_t[i].data, sizeof(float), sz, f) != sz) { fclose(f); return -4; }
    }
    fclose(f); return 0;
}
static float *W(const char *fmt, ...) {
    char name[256]; va_list ap; va_start(ap, fmt); vsnprintf(name, sizeof name, fmt, ap); va_end(ap);
    for (int i = 0; i < g_nt; i++) if (!strcmp(g_t[i].name, name)) return g_t[i].data;
    fprintf(stderr, "reuse_engine: missing weight %s\n", name); exit(1);
}

// ---------- NEON math ----------
// 2^t for t in [-125, 125]: n = round(t) lands in z's low mantissa bits (shift trick), degree-4 poly for 2^r
// (rel err 8e-6, proportional to |r|, so slow decays near 1 stay exact), exponent added as an integer.
// Caller guarantees -125 <= t <= 125 (n must fit the exponent field). The asm barriers stop -ffast-math from folding (t + S) - S to t.
static inline float32x4_t v_exp2_fast(float32x4_t t) {
    float32x4_t z = vaddq_f32(t, vdupq_n_f32(0x1.8p23f)); __asm__("" : "+w"(z));
    float32x4_t n = vsubq_f32(z, vdupq_n_f32(0x1.8p23f)); __asm__("" : "+w"(n));
    float32x4_t r = vsubq_f32(t, n);
    float32x4_t q = vfmaq_f32(vdupq_n_f32(0x1.3d273cp-7f), vdupq_n_f32(0x1.564378p-10f), r);
    q = vfmaq_f32(vdupq_n_f32(0x1.c6ca50p-5f), q, r);
    q = vfmaq_f32(vdupq_n_f32(0x1.ebf8f2p-3f), q, r);
    q = vfmaq_f32(vdupq_n_f32(0x1.62e40ap-1f), q, r);
    float32x4_t p = vfmaq_f32(vdupq_n_f32(1.f), q, r);
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32(vreinterpretq_s32_f32(z), 23)));
}
static inline float32x4_t v_exp2_clamp(float32x4_t t) { return v_exp2_fast(vmaxq_f32(t, vdupq_n_f32(-125.f))); }
// 1/x by estimate + 2 Newton steps (~1 ulp): fdiv has low throughput on one pipe.
static inline float32x4_t v_recip(float32x4_t x) {
    float32x4_t r = vrecpeq_f32(x); r = vmulq_f32(r, vrecpsq_f32(x, r)); return vmulq_f32(r, vrecpsq_f32(x, r));
}
static inline float32x4_t v_silu(float32x4_t x) {  // x * sigmoid(x); e^-x via exp2, clamped to [-125, 125]
    float32x4_t t = vminq_f32(vmaxq_f32(vmulq_n_f32(x, -(float)M_LOG2E), vdupq_n_f32(-125.f)), vdupq_n_f32(125.f));
    return vmulq_f32(x, v_recip(vaddq_f32(vdupq_n_f32(1.f), v_exp2_fast(t))));
}
// softplus(x) = max(x,0) + log1p(exp(-|x|)); log1p(t) = 2 atanh(q), q = t/(2+t) <= 1/3 (series to q^13, ~1e-8 rel)
static inline float32x4_t v_softplus(float32x4_t x) {
    float32x4_t t = v_exp2_clamp(vmulq_n_f32(vabsq_f32(x), -(float)M_LOG2E));
    float32x4_t q = vmulq_f32(t, v_recip(vaddq_f32(vdupq_n_f32(2.f), t))), z = vmulq_f32(q, q);
    float32x4_t pl = vdupq_n_f32(1.f / 13);
    pl = vfmaq_f32(vdupq_n_f32(1.f / 11), pl, z); pl = vfmaq_f32(vdupq_n_f32(1.f / 9), pl, z);
    pl = vfmaq_f32(vdupq_n_f32(1.f / 7), pl, z);  pl = vfmaq_f32(vdupq_n_f32(1.f / 5), pl, z);
    pl = vfmaq_f32(vdupq_n_f32(1.f / 3), pl, z);  pl = vfmaq_f32(vdupq_n_f32(1.f), pl, z);
    float32x4_t sp = vfmaq_f32(vmaxq_f32(x, vdupq_n_f32(0.f)), vaddq_f32(q, q), pl);
    return vbslq_f32(vcgtq_f32(x, vdupq_n_f32(20.f)), x, sp);  // torch threshold 20
}

// ---------- small ops ----------
static inline void gemm(int M, int N, int K, const float *A, int lda, const float *B, int ldb, float *C, int ldc, float beta) {
    PT0();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.f, A, lda, B, ldb, beta, C, ldc);
    PT1(2);
}
static void bias_rows(float *x, int rows, int cols, int ld, const float *b) {
    for (int r = 0; r < rows; r++) memcpy(x + (long)r * ld, b, cols * sizeof(float));
}
// LayerNorm over 64 channels (eps 1e-5, biased var), optional PReLU; one row lives in 16 registers.
static void ln64(float *x, int rows, int ld, const float *w, const float *b, const float *a) {
    PT0();
    for (int r = 0; r < rows; r++) {
        float *v = x + (long)r * ld; float32x4_t q[16], s = vdupq_n_f32(0.f);
        for (int i = 0; i < 16; i++) { q[i] = vld1q_f32(v + 4 * i); s = vaddq_f32(s, q[i]); }
        float32x4_t mean = vdupq_n_f32(vaddvq_f32(s) * (1.f / 64)), ss = vdupq_n_f32(0.f);
        for (int i = 0; i < 16; i++) { q[i] = vsubq_f32(q[i], mean); ss = vfmaq_f32(ss, q[i], q[i]); }
        float32x4_t inv = vdupq_n_f32(1.f / sqrtf(vaddvq_f32(ss) * (1.f / 64) + 1e-5f));
        for (int i = 0; i < 16; i++) {
            float32x4_t y = vfmaq_f32(vld1q_f32(b + 4 * i), vmulq_f32(q[i], inv), vld1q_f32(w + 4 * i));
            if (a) y = vbslq_f32(vcltq_f32(y, vdupq_n_f32(0.f)), vmulq_f32(y, vld1q_f32(a + 4 * i)), y);
            vst1q_f32(v + 4 * i, y);
        }
    }
    PT1(6);
}

// ---------- dense block (4 dilated causal 3x3 convs, 64 ch) ----------
typedef struct {
    int Wd; float *ring, *scratch;    // ring: RING x (Wd+2) x 256, rows 0 and Wd+1 stay zero
    float *w[4], *b[4], *lnw[4], *lnb[4], *pa[4];
} Dense;

static void dense_init(Dense *d, int Wd, const char *pfx) {
    d->Wd = Wd; d->ring = CALLOC((long)RING * (Wd + 2) * 256, sizeof(float)); d->scratch = MALLOC((long)Wd * 64 * 4);
    for (int i = 0; i < 4; i++) {
        d->w[i] = W("%s.dense_block.%d.1.weight", pfx, i); d->b[i] = W("%s.dense_block.%d.1.bias", pfx, i);
        d->lnw[i] = W("%s.dense_block.%d.2.norm.weight", pfx, i); d->lnb[i] = W("%s.dense_block.%d.2.norm.bias", pfx, i);
        d->pa[i] = W("%s.dense_block.%d.3.weight", pfx, i);
    }
}
static inline float *dense_slot(Dense *d, long t) { return d->ring + ((t % RING + RING) % RING) * (long)(d->Wd + 2) * 256; }
static inline float *dense_input(Dense *d, long t) { return dense_slot(d, t) + 256 + 192; }  // row 1, col 192; ld 256
// One dense layer i for rows [r0, r1) of frame t; block input already in dense_input(t) (ld 256). Layer 3 output -> out [Wd][64].
// Layers 0..2 write straight into their concat columns of the current frame (ld 256). A layer reads rows r-1..r+1 of
// earlier layers' columns and writes a column range it does not read, so row chunks of one layer can run in parallel.
static void dense_layer(Dense *d, long t, int i, int r0, int r1, float *out) {
    int n = r1 - r0; if (n <= 0) return;
    float *cur = dense_slot(d, t);
    int dil = 1 << i, c0 = 192 - 64 * i, K = 64 * (i + 1), ldy = (i < 3) ? 256 : 64;
    float *y = ((i < 3) ? cur + 256 + (128 - 64 * i) : out) + (long)r0 * ldy;
    bias_rows(y, n, 64, ldy, d->b[i]);
    for (int kt = 0; kt < 3; kt++) {
        float *src = dense_slot(d, t - (long)(2 - kt) * dil);
        for (int kf = 0; kf < 3; kf++)
            gemm(n, 64, K, src + (long)(r0 + kf) * 256 + c0, 256, d->w[i] + (long)(kt * 3 + kf) * K * 64, 64, y, ldy, 1.f);
    }
    ln64(y, n, ldy, d->lnw[i], d->lnb[i], d->pa[i]);
}

// ---------- mamba ----------
typedef struct { float *A2, *vlim, *D, *inp, *convT, *convb, *xproj, *dtw, *dtb, *outp; } MambaW;
static void mamba_w(MambaW *m, const char *pfx) {
    // A2 = A * log2(e) for exp2; vlim[d] = largest dt for which every state of channel d stays >= 2^-125
    float *A = W("%s.A", pfx); m->A2 = MALLOC(DI * NS * 4); m->vlim = MALLOC(DI * 4);
    for (int d = 0; d < DI; d++) {
        float amax = 0; for (int n = 0; n < NS; n++) { m->A2[d * NS + n] = A[d * NS + n] * (float)M_LOG2E; amax = fmaxf(amax, -m->A2[d * NS + n]); }
        m->vlim[d] = amax > 0 ? 125.f / amax : 1e30f;
    }
    m->D = W("%s.D", pfx); m->inp = W("%s.in_proj.weight", pfx);
    float *cw = W("%s.conv1d.weight", pfx);  // [d][4] -> [4][d]
    m->convT = MALLOC(4 * DI * 4); for (int d = 0; d < DI; d++) for (int k = 0; k < 4; k++) m->convT[k * DI + d] = cw[d * 4 + k];
    m->convb = W("%s.conv1d.bias", pfx); m->xproj = W("%s.x_proj.weight", pfx);
    m->dtw = W("%s.dt_proj.weight", pfx); m->dtb = W("%s.dt_proj.bias", pfx); m->outp = W("%s.out_proj.weight", pfx);
}
typedef struct { float *z, *xc, *dbl, *dt, *y, *tmp; } MScratch;
static void ms_init(MScratch *s, int L) {
    s->z = MALLOC((long)L * DI * 4); s->xc = MALLOC((long)L * DI * 4); s->dbl = MALLOC((long)L * NDBL * 4);
    s->dt = MALLOC((long)L * DI * 4); s->y = MALLOC((long)L * DI * 4); s->tmp = MALLOC((long)L * DI * 4);
}
// xc[r] = silu(bias + sum_k w[k] * src_k[r]); src_k row pointers may be NULL (zero history). ld of sources = DI.
static inline void conv_silu_row(const MambaW *m, const float *s0, const float *s1, const float *s2, const float *s3, float *xc) {
    for (int d = 0; d < DI; d += 4) {
        float32x4_t acc = vld1q_f32(m->convb + d);
        if (s0) acc = vfmaq_f32(acc, vld1q_f32(m->convT + d), vld1q_f32(s0 + d));
        if (s1) acc = vfmaq_f32(acc, vld1q_f32(m->convT + DI + d), vld1q_f32(s1 + d));
        if (s2) acc = vfmaq_f32(acc, vld1q_f32(m->convT + 2 * DI + d), vld1q_f32(s2 + d));
        acc = vfmaq_f32(acc, vld1q_f32(m->convT + 3 * DI + d), vld1q_f32(s3 + d));
        vst1q_f32(xc + d, v_silu(acc));
    }
}
// x_proj, dt_proj, softplus(dt)
static void mamba_proj(MambaW *m, MScratch *s, int L) {
    gemm(L, NDBL, DI, s->xc, DI, m->xproj, NDBL, s->dbl, NDBL, 0.f);
    bias_rows(s->dt, L, DI, DI, m->dtb);
    gemm(L, DI, 4, s->dbl, NDBL, m->dtw, DI, s->dt, DI, 1.f);  // softplus is applied inside the scan
}
// out = y @ out_proj (y is already gated by the scan)
static void mamba_out(MambaW *m, MScratch *s, int L, float *out) {
    gemm(L, 64, DI, s->y, DI, m->outp, 64, out, 64, 0.f);
}
// One recurrence step for row r, channels [d0, d1) (multiples of 4); state h [DI][NS] (row's own state).
// Fused per 4 channels: dt = softplus(raw), decay 2^(dt*A2), state update, readout, + D*u, gate by silu(z).
#define SSM_CH(EXP, k, dd)                                                                                        \
    {                                                                                                             \
        float v = vgetq_lane_f32(v4, k), du = vgetq_lane_f32(du4, k);                                             \
        const float *Ad = A2 + (dd) * NS; float *hd = h + (dd) * NS;                                              \
        float32x4_t h0 = vfmaq_n_f32(vmulq_f32(EXP(vmulq_n_f32(vld1q_f32(Ad), v)), vld1q_f32(hd)), b0, du);       \
        float32x4_t h1 = vfmaq_n_f32(vmulq_f32(EXP(vmulq_n_f32(vld1q_f32(Ad + 4), v)), vld1q_f32(hd + 4)), b1, du); \
        float32x4_t h2 = vfmaq_n_f32(vmulq_f32(EXP(vmulq_n_f32(vld1q_f32(Ad + 8), v)), vld1q_f32(hd + 8)), b2, du); \
        float32x4_t h3 = vfmaq_n_f32(vmulq_f32(EXP(vmulq_n_f32(vld1q_f32(Ad + 12), v)), vld1q_f32(hd + 12)), b3, du); \
        vst1q_f32(hd, h0); vst1q_f32(hd + 4, h1); vst1q_f32(hd + 8, h2); vst1q_f32(hd + 12, h3);                  \
        a[k] = vaddq_f32(vfmaq_f32(vmulq_f32(h0, c0), h1, c1), vfmaq_f32(vmulq_f32(h2, c2), h3, c3));            \
    }
static void ssm_row(const MambaW *m, MScratch *s, int r, float *h, int d0, int d1) {
    PT0();
    const float *raw = s->dt + (long)r * DI, *u = s->xc + (long)r * DI, *z = s->z + (long)r * DI;
    const float *B = s->dbl + (long)r * NDBL + 4, *Cc = B + NS, *A2 = m->A2;
    float32x4_t b0 = vld1q_f32(B), b1 = vld1q_f32(B + 4), b2 = vld1q_f32(B + 8), b3 = vld1q_f32(B + 12);
    float32x4_t c0 = vld1q_f32(Cc), c1 = vld1q_f32(Cc + 4), c2 = vld1q_f32(Cc + 8), c3 = vld1q_f32(Cc + 12);
    float *yr = s->y + (long)r * DI, dtv[DI], gz[DI];
#ifndef NOPROF
    g_prof[3] += (d1 - d0) / 4;
#endif
    // softplus in its own pass: inside the scan loop its latency heads every exp chain and stalls the OoO window (2x slower)
    for (int d = d0; d < d1; d += 4) { vst1q_f32(dtv + d, v_softplus(vld1q_f32(raw + d))); vst1q_f32(gz + d, v_silu(vld1q_f32(z + d))); }
    for (int d = d0; d < d1; d += 4) {
        float32x4_t v4 = vld1q_f32(dtv + d), u4 = vld1q_f32(u + d), du4 = vmulq_f32(v4, u4), a[4];
        if (__builtin_expect(vmaxvq_u32(vcgtq_f32(v4, vld1q_f32(m->vlim + d))) == 0, 1)) {
            SSM_CH(v_exp2_fast, 0, d) SSM_CH(v_exp2_fast, 1, d + 1) SSM_CH(v_exp2_fast, 2, d + 2) SSM_CH(v_exp2_fast, 3, d + 3)
        } else {  // some state would underflow 2^-125: clamp (rare)
#ifndef NOPROF
            g_prof[1] += 1;
#endif
            SSM_CH(v_exp2_clamp, 0, d) SSM_CH(v_exp2_clamp, 1, d + 1) SSM_CH(v_exp2_clamp, 2, d + 2) SSM_CH(v_exp2_clamp, 3, d + 3)
        }
        float32x4_t y4 = vfmaq_f32(vpaddq_f32(vpaddq_f32(a[0], a[1]), vpaddq_f32(a[2], a[3])), vld1q_f32(m->D + d), u4);
        vst1q_f32(yr + d, vmulq_f32(y4, vld1q_f32(gz + d)));
    }
    PT1(0);
}
// in_proj split: x part -> xdst (ld DI), z part -> zdst (ld DI)
static void in_proj(MambaW *m, const float *x, int L, float *xdst, float *zdst) {
    gemm(L, DI, 64, x, 64, m->inp, 2 * DI, xdst, DI, 0.f);
    gemm(L, DI, 64, x, 64, m->inp + DI, 2 * DI, zdst, DI, 0.f);
}

// ---------- thread pool ----------
// threads = N: the calling thread + N-1 persistent workers. pfor(n, f) runs f(0..n-1) and returns when all are done.
// Tasks are claimed with a CAS on a ticket tagged by region number, so a late worker can never run a stale task.
// Workers spin up to 0.3 ms between regions (a hop has ~36 regions), then sleep on a semaphore until the next hop.
#define MAXT 4
#define CH 8   // row chunks per row-parallel region when threads > 1 (tasks are claimed dynamically)
#define NC (e->nc)
#define DCH 4  // channel chunks per direction in the frequency scan (64 channels each)
typedef struct {
    _Atomic uint64_t ticket;  // region << 32 | next task index (0xFFFFFFFF = closed)
    _Atomic int done, sleepers, started;
    int ntask; uint32_t region; void (^fn)(int);
    dispatch_semaphore_t wake;
} Pool;
static Pool g_pool;
static double g_rt_period, g_rt_comp;
void re_set_realtime(double period_ms, double comp_ms) { g_rt_period = period_ms; g_rt_comp = comp_ms; }
static os_workgroup_t g_wg;  // CoreAudio IO workgroup of the output device (live mode), joined by workers
void re_set_workgroup(void *wg) { g_wg = (os_workgroup_t)wg; }

// Put the calling thread in the Mach real-time class (like CoreAudio IO threads): it must finish
// `comp_ms` of work every `period_ms`. Keeps the hop on a performance core when the Mac is busy.
int re_rt_self(double period_ms, double comp_ms) {
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    double ticks_per_ms = 1e6 * tb.denom / tb.numer;
    thread_time_constraint_policy_data_t p = {
        .period = (uint32_t)(period_ms * ticks_per_ms), .computation = (uint32_t)(comp_ms * ticks_per_ms),
        .constraint = (uint32_t)(period_ms * ticks_per_ms), .preemptible = 1};
    return thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
                             (thread_policy_t)&p, THREAD_TIME_CONSTRAINT_POLICY_COUNT);
}
static void pool_claim(Pool *P, uint32_t region) {
    for (;;) {
        uint64_t t = atomic_load(&P->ticket);
        if ((uint32_t)(t >> 32) != region) return;
        uint32_t i = (uint32_t)t;
        if (i >= (uint32_t)P->ntask) return;  // ntask read after the tag check; a reopened ticket fails the CAS below
        if (atomic_compare_exchange_weak(&P->ticket, &t, t + 1)) { P->fn((int)i); atomic_fetch_add(&P->done, 1); }
    }
}
static void *worker_main(void *arg) {
    Pool *P = arg;
    if (g_rt_period > 0) re_rt_self(g_rt_period, g_rt_comp);
    else pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    int joined = 0; os_workgroup_join_token_s tok;
    mach_timebase_info_data_t tb; mach_timebase_info(&tb);
    uint64_t spin_ticks = (uint64_t)(0.3e6 * tb.denom / tb.numer);  // 0.3 ms
    uint32_t seen = 0;
    for (;;) {
        uint64_t t, t0 = mach_absolute_time();
        while ((uint32_t)((t = atomic_load(&P->ticket)) >> 32) == seen) {
            if (mach_absolute_time() - t0 < spin_ticks) continue;
            atomic_fetch_add(&P->sleepers, 1);
            if ((uint32_t)(atomic_load(&P->ticket) >> 32) == seen) dispatch_semaphore_wait(P->wake, DISPATCH_TIME_FOREVER);
            atomic_fetch_sub(&P->sleepers, 1);
            t0 = mach_absolute_time();
        }
        if (!joined && g_wg) { joined = 1; if (os_workgroup_join(g_wg, &tok)) fprintf(stderr, "warning: worker could not join audio workgroup\n"); }
        seen = (uint32_t)(t >> 32);
        pool_claim(P, seen);
    }
    return NULL;
}

// ---------- engine ----------
typedef struct {
    MambaW tm; float *tproj_w, *tproj_b, *tln_w, *tln_b;
    MambaW fw, bw; float *fproj_w, *fproj_b, *fln_w, *fln_b;
    float *conv_hist;  // 4 x Fm x DI ring of in_proj x-part (time direction)
    float *ssm;        // Fm x DI x NS
} Block;
typedef struct {
    float *w, *b, *lnw, *lnb, *pa;              // up_conv1: [3][64][128]
    float *w2, *b2, *lnw2, *lnb2, *pa2;         // up_conv2 (time): [3][64][64]
    float *ring2;                               // 3 x 2Fm x 64
    float *mw, mb, *rw, rb, *iw, ib;            // heads
    float *pad, *up, *o, *ri, *res;
    Dense dense;
} Decoder;
typedef struct { MScratch ms; float *o, *p; } TJob;                 // time-mamba scratch, one per row chunk
typedef struct { MScratch ms; float *so, *xr, *xpart, *h; } FJob;   // frequency-mamba scratch, one per direction
typedef struct {
    int nb, Fin, Fm, layers, threads, nc; long t;  // nc: row chunks per region (1 when single-threaded: bigger sgemm calls)
    float *in_ring;                             // 3 x (Fin+2) x 2
    float *c1w, *c1b, *c1lnw, *c1lnb, *c1pa;
    Dense edense;
    float *c2w, *c2b, *c2lnw, *c2lnb, *c2pa;
    Block blk[12];
    Decoder dmag, dpha;
    TJob tj[CH]; FJob fj[2];
    float *xm, *y3, *cat, *fp;
} Engine;

static void pfor_impl(Engine *e, int n, void (^fn)(int)) {
    if (e->threads < 2 || n < 2) { for (int i = 0; i < n; i++) fn(i); return; }
    Pool *P = &g_pool;
    if (!P->wake) P->wake = dispatch_semaphore_create(0);
    while (atomic_load(&P->started) < e->threads - 1) {
        pthread_t th; pthread_create(&th, NULL, worker_main, P); pthread_detach(th); atomic_fetch_add(&P->started, 1);
    }
    P->fn = fn; P->ntask = n; atomic_store(&P->done, 0);
    uint32_t region = ++P->region;
    atomic_store(&P->ticket, (uint64_t)region << 32);
    for (int s = atomic_load(&P->sleepers); s > 0; s--) dispatch_semaphore_signal(P->wake);
    pool_claim(P, region);
    while (atomic_load(&P->done) < n) {}
    atomic_store(&P->ticket, (uint64_t)region << 32 | 0xFFFFFFFFu);  // close before fn/ntask change
}
#ifndef NOPROF
// per-region profile (profiling build): wall time of each region of a hop and the summed time of its tasks
static double g_rwall[64], g_rwork[64]; static int g_ridx, g_rmax;
void re_prof_regions(double *wall, double *work, int *n) { memcpy(wall, g_rwall, sizeof g_rwall); memcpy(work, g_rwork, sizeof g_rwork); *n = g_rmax; memset(g_rwall, 0, sizeof g_rwall); memset(g_rwork, 0, sizeof g_rwork); }
static void pfor(Engine *e, int n, void (^fn)(int)) {
    int ri = g_ridx++ & 63; if (g_ridx > g_rmax) g_rmax = g_ridx;
    __block _Atomic uint64_t work = 0; uint64_t t0 = mach_absolute_time();
    pfor_impl(e, n, ^(int i) { uint64_t a = mach_absolute_time(); fn(i); atomic_fetch_add(&work, mach_absolute_time() - a); });
    g_rwall[ri] += (mach_absolute_time() - t0) * tb_ms(); g_rwork[ri] += atomic_load(&work) * tb_ms();
}
#else
#define pfor pfor_impl
#endif
// row range start of chunk c (of nc) over n rows
static inline int c0_(int c, int nc, int n) { return (int)((long)n * c / nc); }

static void decoder_init(Decoder *d, int Fm, const char *pfx, int is_mag) {
    char p[128];
    snprintf(p, sizeof p, "%s.dense_block", pfx); dense_init(&d->dense, Fm, p);
    d->w = W("%s.up_conv1.0.conv.weight", pfx); d->b = W("%s.up_conv1.0.conv.bias", pfx);
    d->lnw = W("%s.up_conv1.1.norm.weight", pfx); d->lnb = W("%s.up_conv1.1.norm.bias", pfx); d->pa = W("%s.up_conv1.2.weight", pfx);
    d->w2 = W("%s.up_conv2.0.conv.weight", pfx); d->b2 = W("%s.up_conv2.0.conv.bias", pfx);
    d->lnw2 = W("%s.up_conv2.1.norm.weight", pfx); d->lnb2 = W("%s.up_conv2.1.norm.bias", pfx); d->pa2 = W("%s.up_conv2.2.weight", pfx);
    d->ring2 = CALLOC((long)3 * 2 * Fm * 64, sizeof(float));
    d->pad = CALLOC((long)(Fm + 2) * 64, 4); d->up = CALLOC((long)Fm * 128, 4); d->o = CALLOC((long)2 * Fm * 64, 4);
    d->ri = CALLOC(2 * Fm, 4); d->res = CALLOC(2 * Fm, 4);
    if (is_mag) { d->mw = W("%s.final_conv.weight", pfx); d->mb = W("%s.final_conv.bias", pfx)[0]; }
    else { d->rw = W("%s.phase_conv_r.weight", pfx); d->rb = W("%s.phase_conv_r.bias", pfx)[0];
           d->iw = W("%s.phase_conv_i.weight", pfx); d->ib = W("%s.phase_conv_i.bias", pfx)[0]; }
}

// decoder tail for trunk rows [r0, r1): up_conv1 -> LN/PReLU -> time ring -> up_conv2 -> LN/PReLU -> head. Rows are independent here.
static void decoder_tail(Engine *e, Decoder *d, int r0, int r1, int is_mag) {
    int n = r1 - r0, U = 2 * e->Fm, u0 = 2 * r0, un = 2 * n; if (n <= 0) return;
    float *up = d->up + (long)r0 * 128;                           // [Fm][128] == [2Fm][64]
    bias_rows(up, n, 128, 128, d->b);
    for (int kf = 0; kf < 3; kf++) gemm(n, 128, 64, d->pad + (long)(r0 + kf) * 64, 64, d->w + (long)kf * 64 * 128, 128, up, 128, 1.f);
    float *slot = d->ring2 + (e->t % 3) * (long)U * 64 + (long)u0 * 64;  // LN+PReLU result goes straight into the time ring
    memcpy(slot, up, (long)un * 64 * 4);
    ln64(slot, un, 64, d->lnw, d->lnb, d->pa);
    float *o = d->o + (long)u0 * 64;
    bias_rows(o, un, 64, 64, d->b2);
    for (int kt = 0; kt < 3; kt++)
        gemm(un, 64, 64, d->ring2 + (((e->t - (2 - kt)) % 3 + 3) % 3) * (long)U * 64 + (long)u0 * 64, 64, d->w2 + (long)kt * 64 * 64, 64, o, 64, 1.f);
    ln64(o, un, 64, d->lnw2, d->lnb2, d->pa2);
    float *res = d->res + u0;
    if (is_mag) { cblas_sgemv(CblasRowMajor, CblasNoTrans, un, 64, 1.f, o, 64, d->mw, 1, 0.f, res, 1); for (int r = 0; r < un; r++) res[r] += d->mb; }
    else {
        float *ri = d->ri + u0;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, un, 64, 1.f, o, 64, d->rw, 1, 0.f, ri, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, un, 64, 1.f, o, 64, d->iw, 1, 0.f, res, 1);
        for (int r = 0; r < un; r++) res[r] = atan2f(res[r] + d->ib, ri[r] + d->rb);
    }
}

// time mamba for rows [r0, r1): rows are independent (one recurrent step per frequency bin)
static void time_rows(Engine *e, Block *b, float *x, int r0, int r1, TJob *J) {
    int n = r1 - r0; if (n <= 0) return;
    MScratch *s = &J->ms; long slab = (long)e->Fm * DI;
    float *hk[4]; for (int k = 0; k < 4; k++) hk[k] = b->conv_hist + (((e->t - 3 + k) % 4 + 4) % 4) * slab + (long)r0 * DI;
    float *xs = x + (long)r0 * 64;
    in_proj(&b->tm, xs, n, hk[3], s->z);   // x part written straight into the history ring
    { PT0(); for (int r = 0; r < n; r++) conv_silu_row(&b->tm, hk[0] + r * DI, hk[1] + r * DI, hk[2] + r * DI, hk[3] + r * DI, s->xc + (long)r * DI); PT1(5); }
    mamba_proj(&b->tm, s, n);
    for (int r = 0; r < n; r++) ssm_row(&b->tm, s, r, b->ssm + (long)(r0 + r) * DI * NS, 0, DI);
    float *o = J->o, *p = J->p;
    mamba_out(&b->tm, s, n, o);
    for (int i = 0; i < n * 64; i++) o[i] += xs[i];                 // mamba(x) + x
    bias_rows(p, n, 64, 64, b->tproj_b); gemm(n, 64, 64, o, 64, b->tproj_w, 64, p, 64, 1.f);
    ln64(p, n, 64, b->tln_w, b->tln_b, NULL);
    for (int i = 0; i < n * 64; i++) xs[i] += p[i];                 // TF residual
}

// frequency mamba, both directions; every phase is split into tasks: (direction, row chunk) or (direction, channel chunk)
static void block_run(Engine *e, Block *b, float *x /* [Fm][64] in/out */) {
    int Fm = e->Fm; float *cat = e->cat;  // cat [Fm][128]: fw | bw
    pfor(e, NC, ^(int c) { time_rows(e, b, x, c0_(c, NC, Fm), c0_(c + 1, NC, Fm), &e->tj[c]); });
    MambaW *mfw = &b->fw, *mbw = &b->bw;
#define MW(j) ((j) ? mbw : mfw)
    // 1) input (bw: reversed rows) and in_proj
    pfor(e, 2 * NC, ^(int k) {
        int j = k / NC, r0 = c0_(k % NC, NC, Fm), r1 = c0_(k % NC + 1, NC, Fm), n = r1 - r0; FJob *J = &e->fj[j]; if (n <= 0) return;
        const float *in = x + (long)r0 * 64;
        if (j) { for (int r = r0; r < r1; r++) memcpy(J->xr + (long)r * 64, x + (long)(Fm - 1 - r) * 64, 256); in = J->xr + (long)r0 * 64; }
        in_proj(MW(j), in, n, J->xpart + (long)r0 * DI, J->ms.z + (long)r0 * DI);
    });
    // 2) causal conv over the frequency sequence + SiLU, x_proj, dt_proj
    pfor(e, 2 * NC, ^(int k) {
        int j = k / NC, r0 = c0_(k % NC, NC, Fm), r1 = c0_(k % NC + 1, NC, Fm), n = r1 - r0; FJob *J = &e->fj[j]; if (n <= 0) return;
        MScratch *s = &J->ms; const float *xp = J->xpart;
        { PT0();
          for (int p = r0; p < r1; p++)
              conv_silu_row(MW(j), p >= 3 ? xp + (long)(p - 3) * DI : NULL, p >= 2 ? xp + (long)(p - 2) * DI : NULL,
                            p >= 1 ? xp + (long)(p - 1) * DI : NULL, xp + (long)p * DI, s->xc + (long)p * DI);
          PT1(5); }
        MScratch sub = {s->z + (long)r0 * DI, s->xc + (long)r0 * DI, s->dbl + (long)r0 * NDBL, s->dt + (long)r0 * DI, s->y + (long)r0 * DI, NULL};
        mamba_proj(MW(j), &sub, n);
    });
    // 3) selective scan: sequential over the Fm rows, independent per channel
    pfor(e, 2 * DCH, ^(int k) {
        int j = k / DCH, d0 = (k % DCH) * (DI / DCH), d1 = d0 + DI / DCH; FJob *J = &e->fj[j];
        memset(J->h + (long)d0 * NS, 0, (long)(d1 - d0) * NS * 4);
        for (int p = 0; p < Fm; p++) ssm_row(MW(j), &J->ms, p, J->h, d0, d1);
    });
    // 4) out_proj + residual into the concat buffer (bw rows flipped back)
    pfor(e, 2 * NC, ^(int k) {
        int j = k / NC, r0 = c0_(k % NC, NC, Fm), r1 = c0_(k % NC + 1, NC, Fm), n = r1 - r0; FJob *J = &e->fj[j]; if (n <= 0) return;
        gemm(n, 64, DI, J->ms.y + (long)r0 * DI, DI, MW(j)->outp, 64, J->so + (long)r0 * 64, 64, 0.f);
        if (!j) for (int r = r0; r < r1; r++) for (int c = 0; c < 64; c++) cat[(long)r * 128 + c] = J->so[(long)r * 64 + c] + x[(long)r * 64 + c];
        else for (int r = r0; r < r1; r++) { long q = Fm - 1 - r; for (int c = 0; c < 64; c++) cat[q * 128 + 64 + c] = J->so[(long)r * 64 + c] + J->xr[(long)r * 64 + c]; }
    });
    // 5) output projection + LN + residual
    pfor(e, NC, ^(int c) {
        int r0 = c0_(c, NC, Fm), n = c0_(c + 1, NC, Fm) - r0; if (n <= 0) return;
        float *p = e->fp + (long)r0 * 64;
        bias_rows(p, n, 64, 64, b->fproj_b); gemm(n, 64, 128, cat + (long)r0 * 128, 128, b->fproj_w, 64, p, 64, 1.f);
        ln64(p, n, 64, b->fln_w, b->fln_b, NULL);
        for (int i = 0; i < n * 64; i++) x[(long)r0 * 64 + i] += p[i];
    });
}

// ---------- public API ----------
void *re_create(const char *weights, int nb, int layers) {
    if (!g_t && load_weights(weights)) return NULL;
    if (layers < 1 || layers > 12 || nb < 8) return NULL;
    Engine *e = CALLOC(1, sizeof(Engine));
    e->nb = nb; e->Fin = nb + 2; e->Fm = (e->Fin - 3) / 2 + 1; e->layers = layers;
    int Fin = e->Fin, Fm = e->Fm;
    e->in_ring = CALLOC((long)3 * (Fin + 2) * 2, sizeof(float));
    e->c1w = W("dense_encoder.dense_conv_1_1.0.weight"); e->c1b = W("dense_encoder.dense_conv_1_1.0.bias");
    e->c1lnw = W("dense_encoder.dense_conv_1_1.1.norm.weight"); e->c1lnb = W("dense_encoder.dense_conv_1_1.1.norm.bias");
    e->c1pa = W("dense_encoder.dense_conv_1_1.2.weight");
    dense_init(&e->edense, Fin, "dense_encoder.dense_block");
    e->c2w = W("dense_encoder.dense_conv_2.0.weight"); e->c2b = W("dense_encoder.dense_conv_2.0.bias");
    e->c2lnw = W("dense_encoder.dense_conv_2.1.norm.weight"); e->c2lnb = W("dense_encoder.dense_conv_2.1.norm.bias");
    e->c2pa = W("dense_encoder.dense_conv_2.2.weight");
    for (int l = 0; l < layers; l++) {
        Block *b = &e->blk[l]; char p[96];
        snprintf(p, sizeof p, "TSMamba.%d.time_mamba.forward_blocks", l); mamba_w(&b->tm, p);
        b->tproj_w = W("TSMamba.%d.time_mamba.output_proj.weight", l); b->tproj_b = W("TSMamba.%d.time_mamba.output_proj.bias", l);
        b->tln_w = W("TSMamba.%d.time_mamba.norm.weight", l); b->tln_b = W("TSMamba.%d.time_mamba.norm.bias", l);
        snprintf(p, sizeof p, "TSMamba.%d.freq_mamba.forward_blocks", l); mamba_w(&b->fw, p);
        snprintf(p, sizeof p, "TSMamba.%d.freq_mamba.backward_blocks", l); mamba_w(&b->bw, p);
        b->fproj_w = W("TSMamba.%d.freq_mamba.output_proj.weight", l); b->fproj_b = W("TSMamba.%d.freq_mamba.output_proj.bias", l);
        b->fln_w = W("TSMamba.%d.freq_mamba.norm.weight", l); b->fln_b = W("TSMamba.%d.freq_mamba.norm.bias", l);
        b->conv_hist = CALLOC((long)4 * Fm * DI, sizeof(float)); b->ssm = CALLOC((long)Fm * DI * NS, sizeof(float));
    }
    char p[64];
    snprintf(p, sizeof p, "mask_decoder_list.%d", layers - 1); decoder_init(&e->dmag, Fm, p, 1);
    snprintf(p, sizeof p, "phase_decoder_list.%d", layers - 1); decoder_init(&e->dpha, Fm, p, 0);
    for (int c = 0; c < CH; c++) {  // chunk 0 also serves the single-chunk (1 thread) case
        int tmax = c ? Fm / CH + 2 : Fm;
        ms_init(&e->tj[c].ms, tmax); e->tj[c].o = CALLOC((long)tmax * 64, 4); e->tj[c].p = CALLOC((long)tmax * 64, 4);
    }
    for (int j = 0; j < 2; j++) {
        FJob *J = &e->fj[j]; ms_init(&J->ms, Fm);
        J->so = CALLOC((long)Fm * 64, 4); J->xr = CALLOC((long)Fm * 64, 4); J->xpart = CALLOC((long)Fm * DI, 4); J->h = CALLOC(DI * NS, 4);
    }
    long big = (long)(Fin + 2) * 128;
    e->xm = CALLOC(big, 4); e->y3 = CALLOC(big, 4); e->cat = CALLOC(big, 4); e->fp = CALLOC(big, 4); e->threads = 1; e->nc = 1;
    return e;
}

void re_reset(void *ep) {
    Engine *e = ep; int Fin = e->Fin, Fm = e->Fm; e->t = 0;
    memset(e->in_ring, 0, (long)3 * (Fin + 2) * 2 * 4);
    memset(e->edense.ring, 0, (long)RING * (Fin + 2) * 256 * 4);
    for (int l = 0; l < e->layers; l++) { memset(e->blk[l].conv_hist, 0, (long)4 * Fm * DI * 4); memset(e->blk[l].ssm, 0, (long)Fm * DI * NS * 4); }
    Decoder *ds[2] = {&e->dmag, &e->dpha};
    for (int i = 0; i < 2; i++) { memset(ds[i]->dense.ring, 0, (long)RING * (Fm + 2) * 256 * 4); memset(ds[i]->ring2, 0, (long)3 * 2 * Fm * 64 * 4); }
}

// mag (log1p |X|) and pha (angle) [nb] -> model mag (pre-expm1) and pha [nb]
void re_step(void *ep, const float *mag, const float *pha, float *mag_out, float *pha_out) {
#ifndef NOPROF
    g_ridx = 0;
#endif
    Engine *e = ep; int nb = e->nb, Fin = e->Fin, Fm = e->Fm; long t = e->t;
    // encoder conv1 (2 -> 64, 3x3, causal time with 2 frames of history, freq pad 1); output straight into the dense ring
    float *slot = e->in_ring + (t % 3) * (long)(Fin + 2) * 2;
    memset(slot, 0, (long)(Fin + 2) * 2 * 4);
    for (int f = 0; f < nb; f++) { slot[(f + 1) * 2] = mag[f]; slot[(f + 1) * 2 + 1] = pha[f]; }
    float *x1 = dense_input(&e->edense, t);
    pfor(e, NC, ^(int c) {
        int r0 = c0_(c, NC, Fin), n = c0_(c + 1, NC, Fin) - r0; if (n <= 0) return;
        float *y = x1 + (long)r0 * 256;
        bias_rows(y, n, 64, 256, e->c1b);
        for (int kt = 0; kt < 3; kt++) {
            float *src = e->in_ring + (((t - (2 - kt)) % 3 + 3) % 3) * (long)(Fin + 2) * 2;
            for (int kf = 0; kf < 3; kf++) gemm(n, 64, 2, src + (long)(r0 + kf) * 2, 2, e->c1w + (long)(kt * 3 + kf) * 2 * 64, 64, y, 256, 1.f);
        }
        ln64(y, n, 256, e->c1lnw, e->c1lnb, e->c1pa);
    });
    for (int i = 0; i < 4; i++) pfor(e, NC, ^(int c) { dense_layer(&e->edense, t, i, c0_(c, NC, Fin), c0_(c + 1, NC, Fin), e->y3); });
    // conv2: (1,3) stride 2 over freq, no pad -> Fm rows
    float *x = e->xm;
    pfor(e, NC, ^(int c) {
        int r0 = c0_(c, NC, Fm), n = c0_(c + 1, NC, Fm) - r0; if (n <= 0) return;
        float *y = x + (long)r0 * 64;
        bias_rows(y, n, 64, 64, e->c2b);
        for (int kf = 0; kf < 3; kf++) gemm(n, 64, 64, e->y3 + (long)(2 * r0 + kf) * 64, 128, e->c2w + (long)kf * 64 * 64, 64, y, 64, 1.f);
        ln64(y, n, 64, e->c2lnw, e->c2lnb, e->c2pa);
    });
    for (int l = 0; l < e->layers; l++) block_run(e, &e->blk[l], x);
    // decoders (mag, pha) side by side: task k -> decoder k / NC, row chunk k % NC
    Decoder *dm = &e->dmag, *dp = &e->dpha;
#define DD(k) ((k) < NC ? dm : dp)
    pfor(e, 2 * NC, ^(int k) {
        Decoder *d = DD(k); int r0 = c0_(k % NC, NC, Fm), r1 = c0_(k % NC + 1, NC, Fm);
        float *din = dense_input(&d->dense, t);
        for (int r = r0; r < r1; r++) memcpy(din + (long)r * 256, x + (long)r * 64, 256);
    });
    for (int i = 0; i < 4; i++)  // layer 3 output lands in rows 1..Fm of the zero-padded up_conv1 input
        pfor(e, 2 * NC, ^(int k) { Decoder *d = DD(k); dense_layer(&d->dense, t, i, c0_(k % NC, NC, Fm), c0_(k % NC + 1, NC, Fm), d->pad + 64); });
    pfor(e, 2 * NC, ^(int k) { decoder_tail(e, DD(k), c0_(k % NC, NC, Fm), c0_(k % NC + 1, NC, Fm), k < NC); });
    memcpy(mag_out, e->dmag.res, nb * 4); memcpy(pha_out, e->dpha.res, nb * 4);
    e->t++;
}

void re_set_threads(void *ep, int n) { Engine *e = ep; e->threads = n < 1 ? 1 : n > MAXT ? MAXT : n; e->nc = e->threads > 1 ? CH : 1; }
int re_dims(void *ep, int *Fin, int *Fm) { Engine *e = ep; *Fin = e->Fin; *Fm = e->Fm; return e->nb; }

// ---------- full hop pipeline (matches reuse_core.LiveEnhancer.process, look-ahead 0) ----------
// hop samples in -> hop samples out, output lags input by one hop.
// Real DFT: vDSP FFT when n_fft = f * 2^k (f in 1, 3, 5, 15; true for 8/12/16/24/32/48 kHz), else precomputed matrices
// (44.1/22.05 kHz; at 48 kHz the matrices would be 29 MB of pinned memory and a 29 MB read per hop).
typedef struct {
    void *eng; int sr, hop, nfft, nb, enhanced;
    vDSP_DFT_Setup fft_f, fft_i; float *ev, *od, *Or, *Oi;
    float *win, *env, *inbuf, *tail, *fwd_c, *fwd_s, *inv_c, *inv_s, *frame, *re, *im, *mag, *pha, *mo, *po, *y;
} Live;

static int make_even(int v) { return v % 2 ? v + 1 : v; }

void *re_live_create(const char *weights, int sr, int layers) {
    Live *L = CALLOC(1, sizeof(Live));
    L->sr = sr; L->hop = make_even(160 * sr / 8000); L->nfft = make_even(320 * sr / 8000); L->nb = L->nfft / 2 + 1;
    if (L->nfft != 2 * L->hop) { free(L); return NULL; }
    L->eng = re_create(weights, L->nb, layers); if (!L->eng) { free(L); return NULL; }
    int N = L->nfft, H = L->hop, nb = L->nb; L->enhanced = 1;
    L->win = MALLOC(N * 4); L->env = MALLOC(H * 4);
    for (int n = 0; n < N; n++) L->win[n] = (float)(0.5 - 0.5 * cos(2 * M_PI * n / N));  // torch.hann_window (periodic)
    for (int n = 0; n < H; n++) L->env[n] = L->win[n] * L->win[n] + L->win[n + H] * L->win[n + H];
    L->fft_f = vDSP_DFT_zrop_CreateSetup(NULL, N, vDSP_DFT_FORWARD);
    L->fft_i = L->fft_f ? vDSP_DFT_zrop_CreateSetup(L->fft_f, N, vDSP_DFT_INVERSE) : NULL;
    if (L->fft_i) { L->ev = CALLOC(N / 2, 4); L->od = CALLOC(N / 2, 4); L->Or = CALLOC(N / 2, 4); L->Oi = CALLOC(N / 2, 4); }
    else {
    L->fwd_c = MALLOC((long)nb * N * 4); L->fwd_s = MALLOC((long)nb * N * 4);
    L->inv_c = MALLOC((long)N * nb * 4); L->inv_s = MALLOC((long)N * nb * 4);
    for (int k = 0; k < nb; k++) for (int n = 0; n < N; n++) {
        double a = 2 * M_PI * (double)((long)k * n % N) / N, ck = (k == 0 || 2 * k == N) ? 1.0 : 2.0;
        long m = (long)k * n % N; double sn = (2 * m % N == 0) ? 0.0 : sin(a);  // exact zeros at multiples of pi
        L->fwd_c[(long)k * N + n] = (float)cos(a); L->fwd_s[(long)k * N + n] = (float)-sn;
        L->inv_c[(long)n * nb + k] = (float)(ck * cos(a) / N); L->inv_s[(long)n * nb + k] = (float)(-ck * sn / N);
    }
    }
    float **b[] = {&L->inbuf, &L->tail, &L->frame, &L->re, &L->im, &L->mag, &L->pha, &L->mo, &L->po, &L->y};
    for (unsigned i = 0; i < sizeof b / sizeof *b; i++) *b[i] = CALLOC(N, 4);
    return L;
}
int re_live_hop(void *lp) { return ((Live *)lp)->hop; }
void re_live_set_enhanced(void *lp, int on) { ((Live *)lp)->enhanced = on; }
void re_live_reset(void *lp) {
    Live *L = lp; re_reset(L->eng); memset(L->inbuf, 0, L->hop * 4); memset(L->tail, 0, L->hop * 4);
}
void re_live_process(void *lp, const float *x, float *out) {
    Live *L = lp; int N = L->nfft, H = L->hop, nb = L->nb;
    for (int n = 0; n < H; n++) { L->frame[n] = L->inbuf[n] * L->win[n]; L->frame[n + H] = x[n] * L->win[n + H]; }
    memcpy(L->inbuf, x, H * 4);
    if (L->fft_f) {  // zrop: even/odd samples in, packed out (DC real in Or[0], Nyquist real in Oi[0]), scaled by 2
        DSPSplitComplex z = {L->ev, L->od}; vDSP_ctoz((const DSPComplex *)L->frame, 2, &z, 1, N / 2);
        vDSP_DFT_Execute(L->fft_f, L->ev, L->od, L->Or, L->Oi);
        float h = 0.5f; vDSP_vsmul(L->Or, 1, &h, L->re, 1, N / 2); vDSP_vsmul(L->Oi, 1, &h, L->im, 1, N / 2);
        L->re[N / 2] = L->im[0]; L->im[0] = 0.f; L->im[N / 2] = 0.f;
    } else {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, nb, N, 1.f, L->fwd_c, N, L->frame, 1, 0.f, L->re, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, nb, N, 1.f, L->fwd_s, N, L->frame, 1, 0.f, L->im, 1);
    }
    for (int k = 0; k < nb; k++) { L->mag[k] = log1pf(sqrtf(L->re[k] * L->re[k] + L->im[k] * L->im[k])); L->pha[k] = atan2f(L->im[k], L->re[k]); }
    re_step(L->eng, L->mag, L->pha, L->mo, L->po);  // always run: keeps model state warm for raw/enhanced A/B
    if (!L->enhanced) {  // raw: same one-hop delay, no processing
        memcpy(out, L->tail, H * 4); memcpy(L->tail, x, H * 4); return;
    }
    int zeros = 0;
    for (int k = 0; k < nb; k++) { float m = L->mo[k] > 0.f ? expm1f(L->mo[k]) : 0.f; L->mo[k] = m; zeros += (m == 0.f); }
    if (zeros * 2 > nb) memset(L->mo, 0, nb * 4);  // NVIDIA sweep-artifact guard
    for (int k = 0; k < nb; k++) { L->re[k] = L->mo[k] * cosf(L->po[k]); L->im[k] = L->mo[k] * sinf(L->po[k]); }
    if (L->fft_i) {  // packed in, even/odd out, scaled by N (irfft = 1/N): fold 1/N into the window below
        memcpy(L->ev, L->re, N / 2 * 4); memcpy(L->od, L->im, N / 2 * 4); L->od[0] = L->re[N / 2];
        vDSP_DFT_Execute(L->fft_i, L->ev, L->od, L->Or, L->Oi);
        DSPSplitComplex z = {L->Or, L->Oi}; vDSP_ztoc(&z, 1, (DSPComplex *)L->y, 2, N / 2);
        float s = 1.f / N; vDSP_vsmul(L->y, 1, &s, L->y, 1, N);
    } else {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, nb, 1.f, L->inv_c, nb, L->re, 1, 0.f, L->y, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, N, nb, 1.f, L->inv_s, nb, L->im, 1, 1.f, L->y, 1);
    }
    for (int n = 0; n < N; n++) L->y[n] *= L->win[n];
    for (int n = 0; n < H; n++) out[n] = (L->tail[n] + L->y[n]) / L->env[n];
    memcpy(L->tail, L->y + H, H * 4);
}
void re_live_set_threads(void *lp, int n) { re_set_threads(((Live *)lp)->eng, n); }
