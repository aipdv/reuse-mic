#include <arm_neon.h>
#include <stdio.h>
#include <mach/mach_time.h>
int main(void){ mach_timebase_info_data_t tb; mach_timebase_info(&tb);
 float32x4_t a[16]; for(int i=0;i<16;i++)a[i]=vdupq_n_f32(i*1e-3f); float32x4_t m=vdupq_n_f32(0.9999f), c=vdupq_n_f32(1e-6f);
 long N=200000000/16; uint64_t t0=mach_absolute_time();
 for(long k=0;k<N;k++){ for(int i=0;i<16;i++) a[i]=vfmaq_f32(a[i],c,m); __asm__("":"+w"(a[0]),"+w"(a[8])); }
 double ns=(mach_absolute_time()-t0)*(double)tb.numer/tb.denom; float s=0; for(int i=0;i<16;i++)s+=vaddvq_f32(a[i]);
 printf("%.3f ns per vector fma (%.1f G fma-vectors/s) %g\n", ns/(N*16.0), N*16.0/ns, s); }
