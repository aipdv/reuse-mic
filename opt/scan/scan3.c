// Same math as reuse-mic/shim/mamba_ssm/scan.c, restructured for Apple CPUs:
// exp(dt*A) for a whole time step is computed in one Accelerate vvexpf call (vectorized),
// then the recurrence runs as plain FMA loops. Optional block split over `g_workers` GCD jobs.
#include <math.h>
#include <stdlib.h>
#include <dispatch/dispatch.h>
#include <Accelerate/Accelerate.h>

static int g_workers = 1;
void set_workers(int w) { g_workers = w < 1 ? 1 : w; }
static inline float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
#define QUEUE dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0)

static void run(size_t jobs, void (^body)(size_t, size_t)) {  // body(first, last)
    size_t w = (size_t)g_workers < jobs ? (size_t)g_workers : jobs;
    if (w <= 1) { body(0, jobs); return; }
    size_t per = (jobs + w - 1) / w;
    dispatch_apply(w, QUEUE, ^(size_t k) {
        size_t a = k * per, b = a + per > jobs ? jobs : a + per;
        if (a < b) body(a, b);
    });
}

// dt, u: [b, L, d] ; A: [d, n] ; B, C: [b, L, n] ; y: [b, L, d]
void selective_scan(const float *dt, const float *u, const float *A, const float *B,
                    const float *C, const float *Dskip, float *y, int b, int L, int d, int n) {
    run((size_t)b, ^(size_t b0, size_t b1) {
        int dn = d * n;
        float *h = calloc(dn, sizeof(float)), *e = malloc(dn * sizeof(float)), *sp = malloc(d * sizeof(float));
        for (size_t bi = b0; bi < b1; bi++) {
            for (int i = 0; i < dn; i++) h[i] = 0.f;
            for (int t = 0; t < L; t++) {
                long row = (long)bi * L + t;
                const float *dtr = dt + row * d, *ur = u + row * d, *Bt = B + row * n, *Ct = C + row * n;
                for (int di = 0; di < d; di++) {
                    float s = softplus(dtr[di]); sp[di] = s;
                    for (int k = 0; k < n; k++) e[di * n + k] = s * A[di * n + k];
                }
                vvexpf(e, e, &dn);
                for (int di = 0; di < d; di++) {
                    float du = sp[di] * ur[di], acc = 0.f, *hd = h + di * n; const float *ed = e + di * n;
                    for (int k = 0; k < n; k++) { hd[k] = ed[k] * hd[k] + du * Bt[k]; acc += hd[k] * Ct[k]; }
                    y[row * d + di] = acc + Dskip[di] * ur[di];
                }
            }
        }
        free(h); free(e); free(sp);
    });
}

// dt, x: [b, d] ; B, C: [b, n] ; state: [b, d, n] in/out ; y: [b, d]
void selective_step(const float *dt, const float *x, const float *A, const float *B,
                    const float *C, const float *Dskip, float *state, float *y, int b, int d, int n) {
    run((size_t)b, ^(size_t b0, size_t b1) {
        int dn = d * n;
        float *e = malloc(dn * sizeof(float)), *sp = malloc(d * sizeof(float));
        for (size_t bi = b0; bi < b1; bi++) {
            const float *dtr = dt + bi * d, *xr = x + bi * d, *Bt = B + bi * n, *Ct = C + bi * n;
            float *hb = state + (long)bi * dn;
            for (int di = 0; di < d; di++) {
                float s = softplus(dtr[di]); sp[di] = s;
                for (int k = 0; k < n; k++) e[di * n + k] = s * A[di * n + k];
            }
            vvexpf(e, e, &dn);
            for (int di = 0; di < d; di++) {
                float dx = sp[di] * xr[di], acc = 0.f, *hd = hb + di * n; const float *ed = e + di * n;
                for (int k = 0; k < n; k++) { hd[k] = ed[k] * hd[k] + dx * Bt[k]; acc += hd[k] * Ct[k]; }
                y[bi * d + di] = acc + Dskip[di] * xr[di];
            }
        }
        free(e); free(sp);
    });
}
