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
    int W = g_workers < 1 ? 1 : g_workers, per = (d + W - 1) / W;
    void (^job)(size_t) = ^(size_t j) {
        int bi = (int)(j / W), d0 = (int)(j % W) * per, d1 = d0 + per > d ? d : d0 + per, dd = d1 - d0;
        if (dd <= 0) return;
        int m = dd * n;
        float *h = calloc(m, sizeof(float)), *e = malloc(m * sizeof(float)), *sp = malloc(dd * sizeof(float));
        for (int t = 0; t < L; t++) {
            long row = (long)bi * L + t;
            const float *dtr = dt + row * d, *ur = u + row * d, *Bt = B + row * n, *Ct = C + row * n;
            for (int i = 0; i < dd; i++) {
                float s = softplus(dtr[d0 + i]); sp[i] = s;
                for (int k = 0; k < n; k++) e[i * n + k] = s * A[(d0 + i) * n + k];
            }
            vvexpf(e, e, &m);
            for (int i = 0; i < dd; i++) {
                float du = sp[i] * ur[d0 + i], acc = 0.f, *hd = h + i * n; const float *ed = e + i * n;
                for (int k = 0; k < n; k++) { hd[k] = ed[k] * hd[k] + du * Bt[k]; acc += hd[k] * Ct[k]; }
                y[row * d + d0 + i] = acc + Dskip[d0 + i] * ur[d0 + i];
            }
        }
        free(h); free(e); free(sp);
    };
    size_t jobs = (size_t)b * W;
    if (jobs == 1) job(0); else dispatch_apply(jobs, QUEUE, job);
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
