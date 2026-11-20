// Times the engine's own ssm_row on synthetic data. clang -O3 -ffast-math -mcpu=apple-m1 -DNOPROF lab/ssm_harness.c -framework Accelerate
#include "../reuse_engine.c"
static float fr(void) { return rand() / (float)RAND_MAX; }
static double ms(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return mach_absolute_time() * (double)tb.numer / tb.denom / 1e6; }
int main(int argc, char **argv) {
    int L = 482; MambaW m = {0}; MScratch s; ms_init(&s, L);
    m.A2 = malloc(DI * NS * 4); m.vlim = malloc(DI * 4); m.D = malloc(DI * 4);
    for (int d = 0; d < DI; d++) { float amax = 0; for (int n = 0; n < NS; n++) { float a = -expf(-2.f + 6.3f * fr()) * (float)M_LOG2E; m.A2[d * NS + n] = a; amax = fmaxf(amax, -a); } m.vlim[d] = 125.f / amax; m.D[d] = fr(); }
    for (long i = 0; i < (long)L * DI; i++) { s.dt[i] = -6 + 4 * fr(); s.xc[i] = 2 * fr() - 1; s.z[i] = 4 * fr() - 2; }
    for (long i = 0; i < (long)L * NDBL; i++) s.dbl[i] = 2 * fr() - 1;
    float *h = calloc(DI * NS, 4), *H = calloc((long)L * DI * NS, 4);
    double best = 1e9, best2 = 1e9;
    for (int it = 0; it < 30; it++) {
        memset(h, 0, DI * NS * 4); double t0 = ms();
        for (int r = 0; r < L; r++) ssm_row(&m, &s, r, h, 0, DI);           // freq style: one state, L steps
        best = fmin(best, ms() - t0); t0 = ms();
        for (int r = 0; r < L; r++) ssm_row(&m, &s, r, H + (long)r * DI * NS, 0, DI);  // time style: own state per row
        best2 = fmin(best2, ms() - t0);
    }
    printf("freq-style %.3f ms (%.2f ns/ch-step), time-style %.3f ms (%.2f ns/ch-step)\n", best, best * 1e6 / (L * DI), best2, best2 * 1e6 / (L * DI));
}
