// Scan kernel microbenchmark: one Mamba recurrence over L rows, DI channels, NS states.
// clang -O3 -ffast-math -mcpu=apple-m1 scan_bench.c -o scan_bench && ./scan_bench
#include <arm_neon.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>
#define DI 256
#define NS 16
#define L 482

static double now_ms(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return mach_absolute_time() * (double)tb.numer / tb.denom / 1e6; }

// ---- v0: engine today (expf Cody-Waite, degree 5, rndn + cvt + scale) ----
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
static void v0(const float *A, const float *dt, const float *u, const float *Bm, const float *Cm, const float *D, float *h, float *y) {
    for (int r = 0; r < L; r++) {
        const float *sp = dt + r * DI, *ur = u + r * DI, *B = Bm + r * NS, *C = Cm + r * NS; float *yr = y + r * DI;
        float32x4_t b0 = vld1q_f32(B), b1 = vld1q_f32(B + 4), b2 = vld1q_f32(B + 8), b3 = vld1q_f32(B + 12);
        float32x4_t c0 = vld1q_f32(C), c1 = vld1q_f32(C + 4), c2 = vld1q_f32(C + 8), c3 = vld1q_f32(C + 12);
        for (int d = 0; d < DI; d++) {
            float v = sp[d], du = v * ur[d]; const float *Ad = A + d * NS; float *hd = h + d * NS;
            float32x4_t e0 = v_expf_neg(vmulq_n_f32(vld1q_f32(Ad), v)), e1 = v_expf_neg(vmulq_n_f32(vld1q_f32(Ad + 4), v));
            float32x4_t e2 = v_expf_neg(vmulq_n_f32(vld1q_f32(Ad + 8), v)), e3 = v_expf_neg(vmulq_n_f32(vld1q_f32(Ad + 12), v));
            float32x4_t h0 = vfmaq_n_f32(vmulq_f32(e0, vld1q_f32(hd)), b0, du), h1 = vfmaq_n_f32(vmulq_f32(e1, vld1q_f32(hd + 4)), b1, du);
            float32x4_t h2 = vfmaq_n_f32(vmulq_f32(e2, vld1q_f32(hd + 8)), b2, du), h3 = vfmaq_n_f32(vmulq_f32(e3, vld1q_f32(hd + 12)), b3, du);
            vst1q_f32(hd, h0); vst1q_f32(hd + 4, h1); vst1q_f32(hd + 8, h2); vst1q_f32(hd + 12, h3);
            float32x4_t acc = vmulq_f32(h0, c0); acc = vfmaq_f32(acc, h1, c1); acc = vfmaq_f32(acc, h2, c2); acc = vfmaq_f32(acc, h3, c3);
            yr[d] = vaddvq_f32(acc) + D[d] * ur[d];
        }
    }
}

// ---- exp2 for t = A*log2e*dt <= 0: shift trick (no rndn/cvt), Horner degree 5 or 4 ----
// 2^t = 2^n * 2^r, n = round(t), r in [-0.5, 0.5]. Coefficients: Arm optimized-routines exp2f (1.96 ulp).
#define SHIFT 0x1.8p23f
static inline float32x4_t v_exp2_neg5(float32x4_t t) {
    t = vmaxq_f32(t, vdupq_n_f32(-126.f));
    float32x4_t n = vrndnq_f32(t), r = vsubq_f32(t, n);
    float32x4_t q = vfmaq_f32(vdupq_n_f32(0x1.3ce9e4p-7f), vdupq_n_f32(0x1.59977ap-10f), r);
    q = vfmaq_f32(vdupq_n_f32(0x1.c6bd32p-5f), q, r);
    q = vfmaq_f32(vdupq_n_f32(0x1.ebf9bcp-3f), q, r);
    q = vfmaq_f32(vdupq_n_f32(0x1.62e422p-1f), q, r);
    float32x4_t s = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23));
    return vfmaq_f32(s, vmulq_f32(q, r), s);
}
static inline float32x4_t v_exp2_neg4(float32x4_t t) {  // degree 4 (Taylor-ish minimax), ~4e-6 rel
    t = vmaxq_f32(t, vdupq_n_f32(-126.f));
    float32x4_t n = vrndnq_f32(t), r = vsubq_f32(t, n);
    float32x4_t q = vfmaq_f32(vdupq_n_f32(0x1.caa4fap-5f), vdupq_n_f32(0x1.3b7622p-7f), r);
    q = vfmaq_f32(vdupq_n_f32(0x1.ebff60p-3f), q, r);
    q = vfmaq_f32(vdupq_n_f32(0x1.62df2ap-1f), q, r);
    float32x4_t s = vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23));
    return vfmaq_f32(s, vmulq_f32(q, r), s);
}


// exp2 via shift trick: n lands in z's low mantissa bits (no frintn/fcvtzs). Needs reassociation off.
static inline float32x4_t v_exp2_shift4(float32x4_t t) {
    t = vmaxq_f32(t, vdupq_n_f32(-126.f));
    float32x4_t z = vaddq_f32(t, vdupq_n_f32(SHIFT)); __asm__("" : "+w"(z));  // opaque: fast-math must not fold (t+S)-S
    float32x4_t n = vsubq_f32(z, vdupq_n_f32(SHIFT)); __asm__("" : "+w"(n));
    float32x4_t r = vsubq_f32(t, n);
    float32x4_t q = vfmaq_f32(vdupq_n_f32(0x1.caa4fap-5f), vdupq_n_f32(0x1.3b7622p-7f), r);
    q = vfmaq_f32(vdupq_n_f32(0x1.ebff60p-3f), q, r);
    q = vfmaq_f32(vdupq_n_f32(0x1.62df2ap-1f), q, r);
    float32x4_t s = vreinterpretq_f32_u32(vaddq_u32(vshlq_n_u32(vreinterpretq_u32_f32(z), 23), vdupq_n_u32(0x3f800000)));
    return vfmaq_f32(s, vmulq_f32(q, r), s);
}


// no clamp (caller guarantees t >= -125), shift constant carries the bias, scale applied by integer add
#define SHIFTB (0x1.8p23f)
static inline float32x4_t v_exp2_fast(float32x4_t t) {
    float32x4_t z = vaddq_f32(t, vdupq_n_f32(SHIFTB)); __asm__("" : "+w"(z));
    float32x4_t n = vsubq_f32(z, vdupq_n_f32(SHIFTB)); __asm__("" : "+w"(n));
    float32x4_t r = vsubq_f32(t, n);
    float32x4_t q = vfmaq_f32(vdupq_n_f32(0x1.caa4fap-5f), vdupq_n_f32(0x1.3b7622p-7f), r);
    q = vfmaq_f32(vdupq_n_f32(0x1.ebff60p-3f), q, r);
    q = vfmaq_f32(vdupq_n_f32(0x1.62df2ap-1f), q, r);
    float32x4_t p = vfmaq_f32(vdupq_n_f32(1.f), q, r);
    return vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(p), vshlq_n_s32(vreinterpretq_s32_f32(z), 23)));
}

#define STEP(EXP, A2, d, v, du)                                                                                  \
    {                                                                                                            \
        const float *Ad = A2 + (d) * NS; float *hd = h + (d) * NS;                                               \
        float32x4_t h0 = vfmaq_n_f32(vmulq_f32(EXP(vmulq_n_f32(vld1q_f32(Ad), v)), vld1q_f32(hd)), b0, du);      \
        float32x4_t h1 = vfmaq_n_f32(vmulq_f32(EXP(vmulq_n_f32(vld1q_f32(Ad + 4), v)), vld1q_f32(hd + 4)), b1, du); \
        float32x4_t h2 = vfmaq_n_f32(vmulq_f32(EXP(vmulq_n_f32(vld1q_f32(Ad + 8), v)), vld1q_f32(hd + 8)), b2, du); \
        float32x4_t h3 = vfmaq_n_f32(vmulq_f32(EXP(vmulq_n_f32(vld1q_f32(Ad + 12), v)), vld1q_f32(hd + 12)), b3, du); \
        vst1q_f32(hd, h0); vst1q_f32(hd + 4, h1); vst1q_f32(hd + 8, h2); vst1q_f32(hd + 12, h3);                 \
        acc = vaddq_f32(vfmaq_f32(vmulq_f32(h0, c0), h1, c1), vfmaq_f32(vmulq_f32(h2, c2), h3, c3));             \
    }

// v1/v2: exp2 kernel, 4 channels per iteration, readout of 4 channels combined with one pairwise add tree
#define KERNEL(name, EXP)                                                                                        \
static void name(const float *A2, const float *dt, const float *u, const float *Bm, const float *Cm, const float *D, float *h, float *y) { \
    for (int r = 0; r < L; r++) {                                                                                \
        const float *sp = dt + r * DI, *ur = u + r * DI, *B = Bm + r * NS, *C = Cm + r * NS; float *yr = y + r * DI; \
        float32x4_t b0 = vld1q_f32(B), b1 = vld1q_f32(B + 4), b2 = vld1q_f32(B + 8), b3 = vld1q_f32(B + 12);     \
        float32x4_t c0 = vld1q_f32(C), c1 = vld1q_f32(C + 4), c2 = vld1q_f32(C + 8), c3 = vld1q_f32(C + 12);     \
        for (int d = 0; d < DI; d += 4) {                                                                        \
            float32x4_t v4 = vld1q_f32(sp + d), u4 = vld1q_f32(ur + d), du4 = vmulq_f32(v4, u4), a0, a1, a2, a3, acc; \
            STEP(EXP, A2, d, vgetq_lane_f32(v4, 0), vgetq_lane_f32(du4, 0)); a0 = acc;                           \
            STEP(EXP, A2, d + 1, vgetq_lane_f32(v4, 1), vgetq_lane_f32(du4, 1)); a1 = acc;                       \
            STEP(EXP, A2, d + 2, vgetq_lane_f32(v4, 2), vgetq_lane_f32(du4, 2)); a2 = acc;                       \
            STEP(EXP, A2, d + 3, vgetq_lane_f32(v4, 3), vgetq_lane_f32(du4, 3)); a3 = acc;                       \
            float32x4_t s = vpaddq_f32(vpaddq_f32(a0, a1), vpaddq_f32(a2, a3));                                  \
            vst1q_f32(yr + d, vfmaq_f32(s, vld1q_f32(D + d), u4));                                               \
        }                                                                                                        \
    }                                                                                                            \
}
KERNEL(v1, v_exp2_neg5)
KERNEL(v2, v_exp2_neg4)
KERNEL(v3, v_exp2_shift4)
KERNEL(v4, v_exp2_fast)


static inline float32x4_t v_exp2_clamp(float32x4_t t) { return v_exp2_fast(vmaxq_f32(t, vdupq_n_f32(-125.f))); }
// v5: guarded. vlim[d] = 125 / max_n |A2[d][n]|: if every dt in the group is below it, no state can underflow the exponent.
static float *g_vlim;
static void v5(const float *A2, const float *dt, const float *u, const float *Bm, const float *Cm, const float *D, float *h, float *y) {
    for (int r = 0; r < L; r++) {
        const float *sp = dt + r * DI, *ur = u + r * DI, *B = Bm + r * NS, *C = Cm + r * NS; float *yr = y + r * DI;
        float32x4_t b0 = vld1q_f32(B), b1 = vld1q_f32(B + 4), b2 = vld1q_f32(B + 8), b3 = vld1q_f32(B + 12);
        float32x4_t c0 = vld1q_f32(C), c1 = vld1q_f32(C + 4), c2 = vld1q_f32(C + 8), c3 = vld1q_f32(C + 12);
        for (int d = 0; d < DI; d += 4) {
            float32x4_t v4 = vld1q_f32(sp + d), u4 = vld1q_f32(ur + d), du4 = vmulq_f32(v4, u4), a0, a1, a2, a3, acc;
            if (__builtin_expect(vmaxvq_u32(vcgtq_f32(v4, vld1q_f32(g_vlim + d))) == 0, 1)) {
                STEP(v_exp2_fast, A2, d, vgetq_lane_f32(v4, 0), vgetq_lane_f32(du4, 0)); a0 = acc;
                STEP(v_exp2_fast, A2, d + 1, vgetq_lane_f32(v4, 1), vgetq_lane_f32(du4, 1)); a1 = acc;
                STEP(v_exp2_fast, A2, d + 2, vgetq_lane_f32(v4, 2), vgetq_lane_f32(du4, 2)); a2 = acc;
                STEP(v_exp2_fast, A2, d + 3, vgetq_lane_f32(v4, 3), vgetq_lane_f32(du4, 3)); a3 = acc;
            } else {
                STEP(v_exp2_clamp, A2, d, vgetq_lane_f32(v4, 0), vgetq_lane_f32(du4, 0)); a0 = acc;
                STEP(v_exp2_clamp, A2, d + 1, vgetq_lane_f32(v4, 1), vgetq_lane_f32(du4, 1)); a1 = acc;
                STEP(v_exp2_clamp, A2, d + 2, vgetq_lane_f32(v4, 2), vgetq_lane_f32(du4, 2)); a2 = acc;
                STEP(v_exp2_clamp, A2, d + 3, vgetq_lane_f32(v4, 3), vgetq_lane_f32(du4, 3)); a3 = acc;
            }
            float32x4_t s = vpaddq_f32(vpaddq_f32(a0, a1), vpaddq_f32(a2, a3));
            vst1q_f32(yr + d, vfmaq_f32(s, vld1q_f32(D + d), u4));
        }
    }
}

static float frand(void) { return rand() / (float)RAND_MAX; }
int main(void) {
    float *A = malloc(DI * NS * 4), *A2 = malloc(DI * NS * 4), *dt = malloc(L * DI * 4), *u = malloc(L * DI * 4);
    float *Bm = malloc(L * NS * 4), *Cm = malloc(L * NS * 4), *D = malloc(DI * 4), *h = malloc(DI * NS * 4);
    float *y0 = malloc(L * DI * 4), *y1 = malloc(L * DI * 4);
    for (int i = 0; i < DI * NS; i++) { A[i] = -expf(-2.f + 6.3f * frand()); A2[i] = A[i] * (float)M_LOG2E; }  // -0.13 .. -73
    for (int i = 0; i < L * DI; i++) { dt[i] = log1pf(expf(-6 + 8 * frand())); u[i] = 2 * frand() - 1; }
    for (int i = 0; i < L * NS; i++) { Bm[i] = 2 * frand() - 1; Cm[i] = 2 * frand() - 1; }
    for (int i = 0; i < DI; i++) D[i] = frand();
    g_vlim = malloc(DI * 4); int slow = 0;
    for (int d = 0; d < DI; d++) { float m = 0; for (int n = 0; n < NS; n++) m = fmaxf(m, -A2[d * NS + n]); g_vlim[d] = 125.f / m; }
    for (int i = 0; i < L * DI; i++) slow += dt[i] > g_vlim[i % DI];
    printf("channel-steps needing the slow path: %.3f%%\n", 100.0 * slow / (L * DI));
    // exp accuracy over the scan's input range
    double e5 = 0, e4 = 0;
    for (int i = 0; i < 2000000; i++) {
        float t = -40.f * frand() * frand(); float32x4_t tv = vdupq_n_f32(t * (float)M_LOG2E); double ref = exp((double)t);
        double a = vgetq_lane_f32(v_exp2_neg5(tv), 0), b = vgetq_lane_f32(v_exp2_fast(tv), 0);
        if (ref > 1e-30) { e5 = fmax(e5, fabs(a - ref) / ref); e4 = fmax(e4, fabs(b - ref) / ref); }
    }
    printf("exp2 rel err: deg5 %.2e, deg4 %.2e\n", e5, e4);
    void (*fn[])(const float *, const float *, const float *, const float *, const float *, const float *, float *, float *) = {v0, v1, v2, v3, v4, v5};
    const char *nm[] = {"v0 engine", "v1 exp2 deg5 x4", "v2 exp2 deg4 x4", "v3 shift deg4 x4", "v4 noclamp intadd", "v5 guarded"};
    for (int k = 0; k < 6; k++) {
        double best = 1e9;
        for (int it = 0; it < 30; it++) {
            memset(h, 0, DI * NS * 4); double t0 = now_ms();
            fn[k](k ? A2 : A, dt, u, Bm, Cm, D, h, k ? y1 : y0);
            best = fmin(best, now_ms() - t0);
        }
        double md = 0, mx = 0; for (int i = 0; i < L * DI; i++) { md = fmax(md, fabs(y1[i] - y0[i])); mx = fmax(mx, fabs(y0[i])); }
        printf("%-18s %.3f ms per %d-row scan (%.2f ns/channel-step)  max|dy| %.2e of %.2e\n", nm[k], best, L, best * 1e6 / (L * DI), k ? md : 0, mx);
    }
}
