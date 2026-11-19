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
// 0 ssm scan, 2 sgemm, 5 mamba conv+silu, 6 layernorm/prelu, 7 gate+softplus
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
// expf, Cody-Waite + degree-5 poly (Arm optimized-routines coefficients), ~1-2 ulp on [-87, 88].
static inline float32x4_t v_expf(float32x4_t x) {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(-87.f)), vdupq_n_f32(88.f));
    float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 0x1.715476p+0f));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f));
    r = vfmsq_f32(r, n, vdupq_n_f32(0x1.7f7d1cp-20f));
    float32x4_t p = vdupq_n_f32(0x1.0e4020p-7f);
    p = vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0x1.555e66p-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0x1.ffffecp-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f), p, r);
    // split 2^n in two factors so n up to 128 does not overflow the exponent field
    int32x4_t ni = vcvtq_s32_f32(n), n1 = vshrq_n_s32(ni, 1), n2 = vsubq_s32(ni, n1);
    float32x4_t s1 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n1, vdupq_n_s32(127)), 23));
    float32x4_t s2 = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n2, vdupq_n_s32(127)), 23));
    return vmulq_f32(vmulq_f32(p, s1), s2);
}
// exp for x <= 0 only (scan decay): one clamp, single scale factor.
static inline float32x4_t v_expf_neg(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.f));
    float32x4_t n = vrndnq_f32(vmulq_n_f32(x, 0x1.715476p+0f));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f));
    r = vfmsq_f32(r, n, vdupq_n_f32(0x1.7f7d1cp-20f));
    float32x4_t p = vdupq_n_f32(0x1.0e4020p-7f);
    p = vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0x1.555e66p-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(0x1.ffffecp-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.f), p, r);
    return vmulq_f32(p, vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23)));
}
static inline float32x4_t v_silu(float32x4_t x) {  // x * sigmoid(x)
    return vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.f), v_expf(vnegq_f32(x))));
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
// block input already in dense_input(t) (ld 256). Output y3 -> out [Wd][64].
static void dense_run(Dense *d, long t, float *out) {
    int Wd = d->Wd; float *cur = dense_slot(d, t);
    for (int i = 0; i < 4; i++) {
        int dil = 1 << i, c0 = 192 - 64 * i, K = 64 * (i + 1);
        // layers 0..2 write straight into their concat columns of the current frame (ld 256)
        float *y = (i < 3) ? cur + 256 + (128 - 64 * i) : out; int ldy = (i < 3) ? 256 : 64;
        bias_rows(y, Wd, 64, ldy, d->b[i]);
        for (int kt = 0; kt < 3; kt++) {
            float *src = dense_slot(d, t - (long)(2 - kt) * dil);
            for (int kf = 0; kf < 3; kf++)
                gemm(Wd, 64, K, src + (long)kf * 256 + c0, 256, d->w[i] + (long)(kt * 3 + kf) * K * 64, 64, y, ldy, 1.f);
        }
        ln64(y, Wd, ldy, d->lnw[i], d->lnb[i], d->pa[i]);
    }
}

// ---------- mamba ----------
typedef struct { float *A, *D, *inp, *convT, *convb, *xproj, *dtw, *dtb, *outp; } MambaW;
static void mamba_w(MambaW *m, const char *pfx) {
    m->A = W("%s.A", pfx); m->D = W("%s.D", pfx); m->inp = W("%s.in_proj.weight", pfx);
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
    gemm(L, DI, 4, s->dbl, NDBL, m->dtw, DI, s->dt, DI, 1.f);
    PT0();
    // softplus(x) = max(x,0) + log1p(exp(-|x|)); log1p(t) = 2 atanh(s), s = t/(2+t) <= 1/3 (series to s^13, ~1e-8 rel)
    for (long i = 0; i < (long)L * DI; i += 4) {
        float32x4_t x = vld1q_f32(s->dt + i), t = v_expf_neg(vnegq_f32(vabsq_f32(x)));
        float32x4_t q = vdivq_f32(t, vaddq_f32(vdupq_n_f32(2.f), t)), z = vmulq_f32(q, q);
        float32x4_t pl = vdupq_n_f32(1.f / 13);
        pl = vfmaq_f32(vdupq_n_f32(1.f / 11), pl, z); pl = vfmaq_f32(vdupq_n_f32(1.f / 9), pl, z);
        pl = vfmaq_f32(vdupq_n_f32(1.f / 7), pl, z);  pl = vfmaq_f32(vdupq_n_f32(1.f / 5), pl, z);
        pl = vfmaq_f32(vdupq_n_f32(1.f / 3), pl, z);  pl = vfmaq_f32(vdupq_n_f32(1.f), pl, z);
        float32x4_t sp = vfmaq_f32(vmaxq_f32(x, vdupq_n_f32(0.f)), vaddq_f32(q, q), pl);
        vst1q_f32(s->dt + i, vbslq_f32(vcgtq_f32(x, vdupq_n_f32(20.f)), x, sp));  // torch threshold 20
    }
    PT1(7);
}
// y *= silu(z) ; out = y @ out_proj
static void mamba_out(MambaW *m, MScratch *s, int L, float *out) {
    PT0();
    for (long i = 0; i < (long)L * DI; i += 4) vst1q_f32(s->y + i, vmulq_f32(vld1q_f32(s->y + i), v_silu(vld1q_f32(s->z + i))));
    PT1(7);
    gemm(L, 64, DI, s->y, DI, m->outp, 64, out, 64, 0.f);
}
// One recurrence step for row r; state h [DI][NS]. Fused exp(dt*A) + update + readout, 16 states in 4 registers.
static void ssm_row(const MambaW *m, MScratch *s, int r, float *h) {
    const float *sp = s->dt + (long)r * DI, *u = s->xc + (long)r * DI, *B = s->dbl + (long)r * NDBL + 4, *Cc = B + NS;
    float32x4_t b0 = vld1q_f32(B), b1 = vld1q_f32(B + 4), b2 = vld1q_f32(B + 8), b3 = vld1q_f32(B + 12);
    float32x4_t c0 = vld1q_f32(Cc), c1 = vld1q_f32(Cc + 4), c2 = vld1q_f32(Cc + 8), c3 = vld1q_f32(Cc + 12);
    float *yr = s->y + (long)r * DI; const float *A = m->A;
    for (int d = 0; d < DI; d++) {
        float v = sp[d], du = v * u[d]; const float *Ad = A + d * NS; float *hd = h + d * NS;
        float32x4_t e0 = v_expf_neg(vmulq_n_f32(vld1q_f32(Ad), v)), e1 = v_expf_neg(vmulq_n_f32(vld1q_f32(Ad + 4), v));
        float32x4_t e2 = v_expf_neg(vmulq_n_f32(vld1q_f32(Ad + 8), v)), e3 = v_expf_neg(vmulq_n_f32(vld1q_f32(Ad + 12), v));
        float32x4_t h0 = vfmaq_n_f32(vmulq_f32(e0, vld1q_f32(hd)), b0, du), h1 = vfmaq_n_f32(vmulq_f32(e1, vld1q_f32(hd + 4)), b1, du);
        float32x4_t h2 = vfmaq_n_f32(vmulq_f32(e2, vld1q_f32(hd + 8)), b2, du), h3 = vfmaq_n_f32(vmulq_f32(e3, vld1q_f32(hd + 12)), b3, du);
        vst1q_f32(hd, h0); vst1q_f32(hd + 4, h1); vst1q_f32(hd + 8, h2); vst1q_f32(hd + 12, h3);
        float32x4_t acc = vmulq_f32(h0, c0); acc = vfmaq_f32(acc, h1, c1); acc = vfmaq_f32(acc, h2, c2); acc = vfmaq_f32(acc, h3, c3);
        yr[d] = vaddvq_f32(acc) + m->D[d] * u[d];
    }
}
// in_proj split: x part -> xdst (ld DI), z part -> s->z
static void in_proj(MambaW *m, MScratch *s, const float *x, int L, float *xdst) {
    gemm(L, DI, 64, x, 64, m->inp, 2 * DI, xdst, DI, 0.f);
    gemm(L, DI, 64, x, 64, m->inp + DI, 2 * DI, s->z, DI, 0.f);
}
// full-sequence Mamba over L rows of x [L][64] -> out [L][64] (no residual)
static void mamba_seq(MambaW *m, MScratch *s, const float *x, int L, float *out, float *h, float *xpart) {
    in_proj(m, s, x, L, xpart);
    { PT0();
      for (int p = 0; p < L; p++)
          conv_silu_row(m, p >= 3 ? xpart + (long)(p - 3) * DI : NULL, p >= 2 ? xpart + (long)(p - 2) * DI : NULL,
                        p >= 1 ? xpart + (long)(p - 1) * DI : NULL, xpart + (long)p * DI, s->xc + (long)p * DI);
      PT1(5); }
    mamba_proj(m, s, L);
    memset(h, 0, DI * NS * sizeof(float));
    { PT0(); for (int p = 0; p < L; p++) ssm_row(m, s, p, h); PT1(0); }
    mamba_out(m, s, L, out);
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
    float *pad, *up, *o, *ri, *res;             // own scratch: the two decoders can run in parallel
    Dense dense;
} Decoder;
typedef struct { MScratch ms; float *o, *p, *so, *xr, *xpart, *h; } Job;  // per-thread scratch
typedef struct {
    int nb, Fin, Fm, layers, threads; long t;
    float *in_ring;                             // 3 x (Fin+2) x 2
    float *c1w, *c1b, *c1lnw, *c1lnb, *c1pa;
    Dense edense;
    float *c2w, *c2b, *c2lnw, *c2lnb, *c2pa;
    Block blk[12];
    Decoder dmag, dpha;
    Job job[2];
    float *xm, *y3, *cat;
} Engine;

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
static os_workgroup_t g_wg;  // CoreAudio IO workgroup of the output device (live mode), joined by the helper
void re_set_workgroup(void *wg) { g_wg = (os_workgroup_t)wg; }
// Second worker for threads == 2: a persistent thread (real-time if the engine was told so).
typedef struct { dispatch_semaphore_t go, done; void (^job)(int); double period_ms, comp_ms; int alive; } Helper;
static void *helper_main(void *arg) {
    Helper *h = arg;
    if (h->period_ms > 0) re_rt_self(h->period_ms, h->comp_ms);
    else pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    int joined = 0; os_workgroup_join_token_s tok;
    for (;;) {
        dispatch_semaphore_wait(h->go, DISPATCH_TIME_FOREVER);
        if (!joined && g_wg) { joined = 1; if (os_workgroup_join(g_wg, &tok)) fprintf(stderr, "warning: helper could not join audio workgroup\n"); }
        h->job(1); dispatch_semaphore_signal(h->done);
    }
    return NULL;
}
static Helper *g_helper;  // one per process is enough (one live engine)
static double g_rt_period, g_rt_comp;
void re_set_realtime(double period_ms, double comp_ms) { g_rt_period = period_ms; g_rt_comp = comp_ms; }
// two independent jobs: f(0) here and f(1) on the helper when threads > 1, else in sequence
static void run2(Engine *e, void (^f)(int)) {
    if (e->threads < 2) { f(0); f(1); return; }
    if (!g_helper) {
        g_helper = CALLOC(1, sizeof(Helper)); g_helper->go = dispatch_semaphore_create(0); g_helper->done = dispatch_semaphore_create(0);
        g_helper->period_ms = g_rt_period; g_helper->comp_ms = g_rt_comp;
        pthread_t th; pthread_create(&th, NULL, helper_main, g_helper); pthread_detach(th);
    }
    g_helper->job = f;  // caller waits below, so the stack block outlives the call
    dispatch_semaphore_signal(g_helper->go);
    f(0);
    dispatch_semaphore_wait(g_helper->done, DISPATCH_TIME_FOREVER);
}

// x [Fm][64] -> head output [2Fm] in d->res
static void decoder_run(Engine *e, Decoder *d, const float *x, int is_mag) {
    int Fm = e->Fm, U = 2 * Fm;
    float *din = dense_input(&d->dense, e->t);
    for (int r = 0; r < Fm; r++) memcpy(din + (long)r * 256, x + (long)r * 64, 256);
    dense_run(&d->dense, e->t, d->pad + 64);  // y3 lands in rows 1..Fm of the zero-padded up_conv1 input
    float *up = d->up;                         // [Fm][128] == [2Fm][64]
    bias_rows(up, Fm, 128, 128, d->b);
    for (int kf = 0; kf < 3; kf++) gemm(Fm, 128, 64, d->pad + (long)kf * 64, 64, d->w + (long)kf * 64 * 128, 128, up, 128, 1.f);
    float *slot = d->ring2 + (e->t % 3) * (long)U * 64;  // LN+PReLU result goes straight into the time ring
    memcpy(slot, up, (long)U * 64 * 4);
    ln64(slot, U, 64, d->lnw, d->lnb, d->pa);
    float *o = d->o;
    bias_rows(o, U, 64, 64, d->b2);
    for (int kt = 0; kt < 3; kt++)
        gemm(U, 64, 64, d->ring2 + (((e->t - (2 - kt)) % 3 + 3) % 3) * (long)U * 64, 64, d->w2 + (long)kt * 64 * 64, 64, o, 64, 1.f);
    ln64(o, U, 64, d->lnw2, d->lnb2, d->pa2);
    float *res = d->res;
    if (is_mag) { cblas_sgemv(CblasRowMajor, CblasNoTrans, U, 64, 1.f, o, 64, d->mw, 1, 0.f, res, 1); for (int r = 0; r < U; r++) res[r] += d->mb; }
    else {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, U, 64, 1.f, o, 64, d->rw, 1, 0.f, d->ri, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, U, 64, 1.f, o, 64, d->iw, 1, 0.f, res, 1);
        for (int r = 0; r < U; r++) res[r] = atan2f(res[r] + d->ib, d->ri[r] + d->rb);
    }
}

// time mamba for rows [r0, r1): rows are independent (one recurrent step per frequency bin)
static void time_rows(Engine *e, Block *b, float *x, int r0, int r1, Job *J) {
    int n = r1 - r0; if (n <= 0) return;
    MScratch *s = &J->ms; long slab = (long)e->Fm * DI;
    float *hk[4]; for (int k = 0; k < 4; k++) hk[k] = b->conv_hist + (((e->t - 3 + k) % 4 + 4) % 4) * slab + (long)r0 * DI;
    float *xs = x + (long)r0 * 64;
    in_proj(&b->tm, s, xs, n, hk[3]);   // x part written straight into the history ring
    { PT0(); for (int r = 0; r < n; r++) conv_silu_row(&b->tm, hk[0] + r * DI, hk[1] + r * DI, hk[2] + r * DI, hk[3] + r * DI, s->xc + (long)r * DI); PT1(5); }
    mamba_proj(&b->tm, s, n);
    { PT0(); for (int r = 0; r < n; r++) ssm_row(&b->tm, s, r, b->ssm + (long)(r0 + r) * DI * NS); PT1(0); }
    float *o = J->o, *p = J->p;
    mamba_out(&b->tm, s, n, o);
    for (int i = 0; i < n * 64; i++) o[i] += xs[i];                 // mamba(x) + x
    bias_rows(p, n, 64, 64, b->tproj_b); gemm(n, 64, 64, o, 64, b->tproj_w, 64, p, 64, 1.f);
    ln64(p, n, 64, b->tln_w, b->tln_b, NULL);
    for (int i = 0; i < n * 64; i++) xs[i] += p[i];                 // TF residual
}

static void block_run(Engine *e, Block *b, float *x /* [Fm][64] in/out */) {
    int Fm = e->Fm, half = (Fm + 1) / 2; float *cat = e->cat;  // cat [Fm][128]
    run2(e, ^(int j) { time_rows(e, b, x, j ? half : 0, j ? Fm : half, &e->job[j]); });
    // ---- freq mamba: forward (job 0) and backward (job 1) over the Fm rows ----
    run2(e, ^(int j) {
        Job *J = &e->job[j];
        if (!j) {
            mamba_seq(&b->fw, &J->ms, x, Fm, J->so, J->h, J->xpart);
            for (int r = 0; r < Fm; r++) for (int c = 0; c < 64; c++) cat[(long)r * 128 + c] = J->so[(long)r * 64 + c] + x[(long)r * 64 + c];
        } else {
            for (int r = 0; r < Fm; r++) memcpy(J->xr + (long)r * 64, x + (long)(Fm - 1 - r) * 64, 256);
            mamba_seq(&b->bw, &J->ms, J->xr, Fm, J->so, J->h, J->xpart);
            for (int r = 0; r < Fm; r++) { long q = Fm - 1 - r; for (int c = 0; c < 64; c++) cat[q * 128 + 64 + c] = J->so[(long)r * 64 + c] + J->xr[(long)r * 64 + c]; }
        }
    });
    float *p = e->job[0].p;
    bias_rows(p, Fm, 64, 64, b->fproj_b); gemm(Fm, 64, 128, cat, 128, b->fproj_w, 64, p, 64, 1.f);
    ln64(p, Fm, 64, b->fln_w, b->fln_b, NULL);
    for (int i = 0; i < Fm * 64; i++) x[i] += p[i];
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
    long big = (long)(Fin + 2) * 128;
    for (int j = 0; j < 2; j++) {
        Job *J = &e->job[j]; ms_init(&J->ms, Fin);
        J->o = CALLOC(big, 4); J->p = CALLOC(big, 4); J->so = CALLOC(big, 4); J->xr = CALLOC(big, 4);
        J->xpart = CALLOC((long)Fin * DI, 4); J->h = CALLOC(DI * NS, 4);
    }
    e->xm = CALLOC(big, 4); e->y3 = CALLOC(big, 4); e->cat = CALLOC(big, 4); e->threads = 1;
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
    Engine *e = ep; int nb = e->nb, Fin = e->Fin, Fm = e->Fm;
    // encoder conv1 (2 -> 64, 3x3, causal time with 2 frames of history, freq pad 1)
    float *slot = e->in_ring + (e->t % 3) * (long)(Fin + 2) * 2;
    memset(slot, 0, (long)(Fin + 2) * 2 * 4);
    for (int f = 0; f < nb; f++) { slot[(f + 1) * 2] = mag[f]; slot[(f + 1) * 2 + 1] = pha[f]; }
    float *x1 = dense_input(&e->edense, e->t);  // conv1 output goes straight into the dense ring (ld 256)
    bias_rows(x1, Fin, 64, 256, e->c1b);
    for (int kt = 0; kt < 3; kt++) {
        float *src = e->in_ring + (((e->t - (2 - kt)) % 3 + 3) % 3) * (long)(Fin + 2) * 2;
        for (int kf = 0; kf < 3; kf++) gemm(Fin, 64, 2, src + kf * 2, 2, e->c1w + (long)(kt * 3 + kf) * 2 * 64, 64, x1, 256, 1.f);
    }
    ln64(x1, Fin, 256, e->c1lnw, e->c1lnb, e->c1pa);
    dense_run(&e->edense, e->t, e->y3);
    // conv2: (1,3) stride 2 over freq, no pad -> Fm rows
    float *x = e->xm;
    bias_rows(x, Fm, 64, 64, e->c2b);
    for (int kf = 0; kf < 3; kf++) gemm(Fm, 64, 64, e->y3 + (long)kf * 64, 128, e->c2w + (long)kf * 64 * 64, 64, x, 64, 1.f);
    ln64(x, Fm, 64, e->c2lnw, e->c2lnb, e->c2pa);
    for (int l = 0; l < e->layers; l++) block_run(e, &e->blk[l], x);
    run2(e, ^(int j) { decoder_run(e, j ? &e->dpha : &e->dmag, x, !j); });
    memcpy(mag_out, e->dmag.res, nb * 4); memcpy(pha_out, e->dpha.res, nb * 4);
    e->t++;
}

void re_set_threads(void *ep, int n) { ((Engine *)ep)->threads = n > 1 ? 2 : 1; }
int re_dims(void *ep, int *Fin, int *Fm) { Engine *e = ep; *Fin = e->Fin; *Fm = e->Fm; return e->nb; }

// ---------- full hop pipeline (matches reuse_core.LiveEnhancer.process, look-ahead 0) ----------
// hop samples in -> hop samples out, output lags input by one hop. Real DFT as precomputed matrices (any n_fft).
typedef struct {
    void *eng; int sr, hop, nfft, nb, enhanced;
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
    L->fwd_c = MALLOC((long)nb * N * 4); L->fwd_s = MALLOC((long)nb * N * 4);
    L->inv_c = MALLOC((long)N * nb * 4); L->inv_s = MALLOC((long)N * nb * 4);
    for (int k = 0; k < nb; k++) for (int n = 0; n < N; n++) {
        double a = 2 * M_PI * (double)((long)k * n % N) / N, ck = (k == 0 || 2 * k == N) ? 1.0 : 2.0;
        long m = (long)k * n % N; double sn = (2 * m % N == 0) ? 0.0 : sin(a);  // exact zeros at multiples of pi
        L->fwd_c[(long)k * N + n] = (float)cos(a); L->fwd_s[(long)k * N + n] = (float)-sn;
        L->inv_c[(long)n * nb + k] = (float)(ck * cos(a) / N); L->inv_s[(long)n * nb + k] = (float)(-ck * sn / N);
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
    cblas_sgemv(CblasRowMajor, CblasNoTrans, nb, N, 1.f, L->fwd_c, N, L->frame, 1, 0.f, L->re, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, nb, N, 1.f, L->fwd_s, N, L->frame, 1, 0.f, L->im, 1);
    for (int k = 0; k < nb; k++) { L->mag[k] = log1pf(sqrtf(L->re[k] * L->re[k] + L->im[k] * L->im[k])); L->pha[k] = atan2f(L->im[k], L->re[k]); }
    re_step(L->eng, L->mag, L->pha, L->mo, L->po);  // always run: keeps model state warm for raw/enhanced A/B
    if (!L->enhanced) {  // raw: same one-hop delay, no processing
        memcpy(out, L->tail, H * 4); memcpy(L->tail, x, H * 4); return;
    }
    int zeros = 0;
    for (int k = 0; k < nb; k++) { float m = L->mo[k] > 0.f ? expm1f(L->mo[k]) : 0.f; L->mo[k] = m; zeros += (m == 0.f); }
    if (zeros * 2 > nb) memset(L->mo, 0, nb * 4);  // NVIDIA sweep-artifact guard
    for (int k = 0; k < nb; k++) { L->re[k] = L->mo[k] * cosf(L->po[k]); L->im[k] = L->mo[k] * sinf(L->po[k]); }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N, nb, 1.f, L->inv_c, nb, L->re, 1, 0.f, L->y, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N, nb, 1.f, L->inv_s, nb, L->im, 1, 1.f, L->y, 1);
    for (int n = 0; n < N; n++) L->y[n] *= L->win[n];
    for (int n = 0; n < H; n++) out[n] = (L->tail[n] + L->y[n]) / L->env[n];
    memcpy(L->tail, L->y + H, H * 4);
}
void re_live_set_threads(void *lp, int n) { re_set_threads(((Live *)lp)->eng, n); }
