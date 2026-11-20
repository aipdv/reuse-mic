// Does sgemm throughput scale with threads (shared AMX per cluster)? Shape = Mamba in_proj chunk at 48 kHz.
#include <Accelerate/Accelerate.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <mach/mach_time.h>
static double ms(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return mach_absolute_time() * (double)tb.numer / tb.denom / 1e6; }
static int M = 482, ITER = 400;
static void *run(void *_) {
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    float *A = calloc(M * 64, 4), *B = calloc(64 * 512, 4), *C = calloc(M * 512, 4);
    for (int i = 0; i < ITER; i++) cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, 512, 64, 1, A, 64, B, 512, 0, C, 512);
    return NULL;
}
int main(void) {
    for (int n = 1; n <= 4; n *= 2) {
        pthread_t th[4]; double t0 = ms();
        for (int i = 0; i < n; i++) pthread_create(&th[i], NULL, run, NULL);
        for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
        double dt = ms() - t0; printf("%d threads: %.1f GFLOPS aggregate\n", n, 2.0 * M * 512 * 64 * ITER * n / dt / 1e6);
    }
}
