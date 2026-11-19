// Same math as reuse-mic/shim/mamba_ssm/scan.c; jobs are split into `g_workers` contiguous
// blocks (1 = plain loop on the calling thread, no GCD wake-ups).
#include <math.h>
#include <dispatch/dispatch.h>

static int g_workers = 1;
void set_workers(int w) { g_workers = w < 1 ? 1 : w; }
static inline float softplus(float x) { return x > 20.f ? x : log1pf(expf(x)); }
#define QUEUE dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0)

static void run(size_t jobs, void (^body)(size_t)) {
    if (g_workers == 1 || jobs < 2) { for (size_t j = 0; j < jobs; j++) body(j); return; }
    size_t w = (size_t)g_workers, per = (jobs + w - 1) / w;
    dispatch_apply(w, QUEUE, ^(size_t k) {
        size_t a = k * per, b = a + per > jobs ? jobs : a + per;
        for (size_t j = a; j < b; j++) body(j);
    });
}

void selective_scan(const float *dt, const float *u, const float *A, const float *B,
                    const float *C, const float *Dskip, float *y, int b, int L, int d, int n) {
    run((size_t)b * d, ^(size_t job) {
        int bi = (int)(job / d), di = (int)(job % d);
        float h[16] = {0};
        const float *Ad = A + di * n;
        for (int t = 0; t < L; t++) {
            long row = ((long)bi * L + t);
            float dtv = softplus(dt[row * d + di]), uv = u[row * d + di];
            const float *Bt = B + row * n, *Ct = C + row * n;
            float acc = 0.f, du = dtv * uv, e[16];
            for (int k = 0; k < n; k++) e[k] = expf(dtv * Ad[k]);
            for (int k = 0; k < n; k++) { h[k] = e[k] * h[k] + du * Bt[k]; acc += h[k] * Ct[k]; }
            y[row * d + di] = acc + Dskip[di] * uv;
        }
    });
}

void selective_step(const float *dt, const float *x, const float *A, const float *B,
                    const float *C, const float *Dskip, float *state, float *y, int b, int d, int n) {
    run((size_t)b, ^(size_t bi) {
        for (int di = 0; di < d; di++) {
            float dtv = softplus(dt[bi * d + di]), xv = x[bi * d + di], dx = dtv * xv, acc = 0.f, e[16];
            const float *Ad = A + di * n, *Bt = B + bi * n, *Ct = C + bi * n;
            float *h = state + ((long)bi * d + di) * n;
            for (int k = 0; k < n; k++) e[k] = expf(dtv * Ad[k]);
            for (int k = 0; k < n; k++) { h[k] = e[k] * h[k] + dx * Bt[k]; acc += h[k] * Ct[k]; }
            y[bi * d + di] = acc + Dskip[di] * xv;
        }
    });
}
