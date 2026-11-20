// reuse: native RE-USE voice cleaner for macOS (no Python). Uses reuse_engine.c.
//   reuse devices
//   reuse live [--input NAME] [--output "BlackHole 2ch"] [--rate 16000] [--layers 4] [--seconds N]
//   reuse file IN OUT [--rate 16000] [--layers 4]
//   reuse bench [--rate 16000] [--layers 4]
// Weights: $REUSE_WEIGHTS, else reuse_rt.bin next to the binary.
#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#include <dispatch/dispatch.h>
#include <mach-o/dyld.h>
#include <mach/mach_time.h>
#include <libgen.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <os/workgroup.h>

void *re_live_create(const char *weights, int sr, int layers);
int re_live_hop(void *lp);
void re_live_process(void *lp, const float *x, float *out);
void re_live_set_enhanced(void *lp, int on);
void re_live_set_threads(void *lp, int n);
int re_rt_self(double period_ms, double comp_ms);
void re_set_realtime(double period_ms, double comp_ms);
void re_set_workgroup(void *wg);

// ---------- args ----------
static const char *arg_s(int argc, char **argv, const char *k, const char *def) {
    for (int i = 0; i < argc - 1; i++) if (!strcmp(argv[i], k)) return argv[i + 1];
    return def;
}
static int arg_i(int argc, char **argv, const char *k, int def) { const char *s = arg_s(argc, argv, k, NULL); return s ? atoi(s) : def; }
static const char *weights_path(void) {
    const char *env = getenv("REUSE_WEIGHTS"); if (env) return env;
    static char buf[PATH_MAX], out[PATH_MAX + 32]; uint32_t n = sizeof buf;
    if (_NSGetExecutablePath(buf, &n) == 0) { char *real = realpath(buf, NULL); snprintf(out, sizeof out, "%s/reuse_rt.bin", dirname(real ? real : buf)); free(real); return out; }
    return "reuse_rt.bin";
}
// default worker count by model rate (M3 Air, 4 layers): the hop must stay well under the 15 ms real-time budget,
// or macOS demotes the thread. 16 kHz: 1 (9 ms), 24/32 kHz: 2 (7/9 ms), 48 kHz: 4 (8.7 ms; 2 threads hit 13 ms and glitch).
static int auto_threads(int rate) { return rate > 32000 ? 4 : rate > 16000 ? 2 : 1; }
static double now_ms(void) { static mach_timebase_info_data_t tb; if (!tb.denom) mach_timebase_info(&tb); return mach_absolute_time() * (double)tb.numer / tb.denom / 1e6; }
#define CK(x) do { OSStatus _s = (x); if (_s) { fprintf(stderr, "CoreAudio error %d at %s:%d (%s)\n", (int)_s, __FILE__, __LINE__, #x); exit(1); } } while (0)

// ---------- devices ----------
static int list_devices(AudioDeviceID **out) {
    AudioObjectPropertyAddress a = {kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    UInt32 sz = 0; AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &a, 0, NULL, &sz);
    *out = malloc(sz); AudioObjectGetPropertyData(kAudioObjectSystemObject, &a, 0, NULL, &sz, *out);
    return (int)(sz / sizeof(AudioDeviceID));
}
static void dev_name(AudioDeviceID d, char *buf, int n) {
    AudioObjectPropertyAddress a = {kAudioObjectPropertyName, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    CFStringRef s = NULL; UInt32 sz = sizeof s; buf[0] = 0;
    if (!AudioObjectGetPropertyData(d, &a, 0, NULL, &sz, &s) && s) { CFStringGetCString(s, buf, n, kCFStringEncodingUTF8); CFRelease(s); }
}
static int dev_channels(AudioDeviceID d, int input) {
    AudioObjectPropertyAddress a = {kAudioDevicePropertyStreamConfiguration, input ? kAudioObjectPropertyScopeInput : kAudioObjectPropertyScopeOutput, kAudioObjectPropertyElementMain};
    UInt32 sz = 0; if (AudioObjectGetPropertyDataSize(d, &a, 0, NULL, &sz) || !sz) return 0;
    AudioBufferList *bl = malloc(sz); int ch = 0;
    if (!AudioObjectGetPropertyData(d, &a, 0, NULL, &sz, bl)) for (UInt32 i = 0; i < bl->mNumberBuffers; i++) ch += bl->mBuffers[i].mNumberChannels;
    free(bl); return ch;
}
static double dev_rate(AudioDeviceID d) {
    AudioObjectPropertyAddress a = {kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    Float64 r = 0; UInt32 sz = sizeof r; AudioObjectGetPropertyData(d, &a, 0, NULL, &sz, &r); return r;
}
static AudioDeviceID default_dev(int input) {
    AudioObjectPropertyAddress a = {input ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    AudioDeviceID d = 0; UInt32 sz = sizeof d; AudioObjectGetPropertyData(kAudioObjectSystemObject, &a, 0, NULL, &sz, &d); return d;
}
static AudioDeviceID find_dev(const char *needle, int input) {
    if (!needle) return default_dev(input);
    AudioDeviceID *ds; int n = list_devices(&ds); AudioDeviceID found = 0; char name[256];
    for (int i = 0; i < n && !found; i++) {
        dev_name(ds[i], name, sizeof name);
        if (strcasestr(name, needle) && dev_channels(ds[i], input) > 0) found = ds[i];
    }
    free(ds); return found;
}
static int cmd_devices(void) {
    AudioDeviceID *ds; int n = list_devices(&ds); char name[256];
    AudioDeviceID di = default_dev(1), dout = default_dev(0);
    for (int i = 0; i < n; i++) {
        dev_name(ds[i], name, sizeof name);
        printf("%-40s in %2d  out %2d  %6.0f Hz%s%s\n", name, dev_channels(ds[i], 1), dev_channels(ds[i], 0), dev_rate(ds[i]),
               ds[i] == di ? "  [default input]" : "", ds[i] == dout ? "  [default output]" : "");
    }
    free(ds); return 0;
}

// ---------- SPSC ring ----------
typedef struct { float *buf; int cap; _Atomic long w, r; } Ring;
static void ring_init(Ring *q, int cap) { q->buf = calloc(cap, 4); q->cap = cap; atomic_init(&q->w, 0); atomic_init(&q->r, 0); }
static long ring_avail(Ring *q) { return atomic_load(&q->w) - atomic_load(&q->r); }
static int ring_write(Ring *q, const float *x, int n) {  // drops on overflow (returns written)
    long w = atomic_load(&q->w), r = atomic_load(&q->r); int space = q->cap - (int)(w - r); if (n > space) n = space;
    for (int i = 0; i < n; i++) q->buf[(w + i) % q->cap] = x[i];
    atomic_store(&q->w, w + n); return n;
}
static int ring_read(Ring *q, float *x, int n) {
    long w = atomic_load(&q->w), r = atomic_load(&q->r); int have = (int)(w - r); if (n > have) n = have;
    for (int i = 0; i < n; i++) x[i] = q->buf[(r + i) % q->cap];
    atomic_store(&q->r, r + n); return n;
}

// ---------- live ----------
typedef struct {
    AudioUnit in_unit, out_unit; Ring in_ring, out_ring; dispatch_semaphore_t sem;
    void *live; int hop, rate, out_ch; double dev_in_rate; AudioConverterRef conv;
    float *render_buf, *conv_src; int conv_src_n, conv_src_pos;
    _Atomic int stop, underruns, overruns; long hops, late, late_at[16]; double *proc_ms; long proc_cap;
    float *tap; long tap_n, tap_cap; const char *feed_path; _Atomic int feed_done; int realtime; os_workgroup_t _Atomic wg;
} App;
static App g;

static OSStatus in_cb(void *ref, AudioUnitRenderActionFlags *flags, const AudioTimeStamp *ts, UInt32 bus, UInt32 frames, AudioBufferList *unused) {
    AudioBufferList bl = {1, {{1, frames * 4, g.render_buf}}};
    OSStatus s = AudioUnitRender(g.in_unit, flags, ts, bus, frames, &bl);
    if (!s) { if (ring_write(&g.in_ring, g.render_buf, (int)frames) < (int)frames) atomic_fetch_add(&g.overruns, 1); dispatch_semaphore_signal(g.sem); }
    return s;
}
static OSStatus out_cb(void *ref, AudioUnitRenderActionFlags *flags, const AudioTimeStamp *ts, UInt32 bus, UInt32 frames, AudioBufferList *io) {
    float tmp[4096]; int n = frames > 4096 ? 4096 : (int)frames;
    int got = ring_read(&g.out_ring, tmp, n);
    if (got < n) { memset(tmp + got, 0, (n - got) * 4); atomic_fetch_add(&g.underruns, 1); }
    float *o = io->mBuffers[0].mData;  // interleaved out_ch channels
    for (int i = 0; i < n; i++) for (int c = 0; c < g.out_ch; c++) o[i * g.out_ch + c] = tmp[i];
    return noErr;
}
static OSStatus conv_proc(AudioConverterRef c, UInt32 *npk, AudioBufferList *io, AudioStreamPacketDescription **d, void *ref) {
    int left = g.conv_src_n - g.conv_src_pos;
    if (left <= 0) { *npk = 0; return 'nomo'; }  // no more input right now; converter keeps its state
    if ((int)*npk > left) *npk = left;
    io->mBuffers[0].mData = g.conv_src + g.conv_src_pos; io->mBuffers[0].mDataByteSize = *npk * 4; io->mBuffers[0].mNumberChannels = 1;
    g.conv_src_pos += *npk; return noErr;
}
static void *worker(void *arg) {
    double hop_ms = 1000.0 * g.hop / g.rate;
    if (g.realtime) { if (re_rt_self(hop_ms, hop_ms * 0.75)) fprintf(stderr, "warning: real-time scheduling refused\n"); }
    else pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // stay on performance cores
    int H = g.hop; float *pend = calloc(H * 64, 4), *y = calloc(H, 4), *conv_out = calloc(H * 64, 4); int npend = 0;
    g.conv_src = calloc(48000, 4);
    int joined = 0; os_workgroup_join_token_s tok;
    while (!atomic_load(&g.stop)) {
        dispatch_semaphore_wait(g.sem, dispatch_time(DISPATCH_TIME_NOW, 200 * NSEC_PER_MSEC));
        os_workgroup_t wg = atomic_load(&g.wg);
        if (!joined && wg) { joined = 1; if (os_workgroup_join(wg, &tok)) { joined = 2; fprintf(stderr, "warning: worker could not join audio workgroup\n"); } }
        g.conv_src_n = ring_read(&g.in_ring, g.conv_src, 48000); g.conv_src_pos = 0;
        if (!g.conv_src_n) continue;
        if (g.conv) {  // device rate -> model rate
            for (;;) {
                UInt32 want = (UInt32)(H * 64 - npend); if (!want) break;
                AudioBufferList bl = {1, {{1, want * 4, conv_out}}};
                OSStatus s = AudioConverterFillComplexBuffer(g.conv, conv_proc, NULL, &want, &bl, NULL);
                memcpy(pend + npend, conv_out, want * 4); npend += want;
                if (s == 'nomo' || want == 0) break;
            }
        } else { int n = g.conv_src_n; if (npend + n > H * 64) n = H * 64 - npend; memcpy(pend + npend, g.conv_src, n * 4); npend += n; }
        int off = 0;
        while (npend - off >= H) {
            double t0 = now_ms(); re_live_process(g.live, pend + off, y); double dt = now_ms() - t0;
            if (g.hops < g.proc_cap) g.proc_ms[g.hops] = dt;
            if (dt > 1000.0 * H / g.rate) { if (g.late < 16) g.late_at[g.late] = g.hops; g.late++; }
            g.hops++;
            ring_write(&g.out_ring, y, H); off += H;
            if (g.tap && g.tap_n + H <= g.tap_cap) { memcpy(g.tap + g.tap_n, y, H * 4); g.tap_n += H; }
        }
        memmove(pend, pend + off, (npend - off) * 4); npend -= off;
    }
    if (joined == 1) os_workgroup_leave(atomic_load(&g.wg), &tok);
    return NULL;
}
static AudioStreamBasicDescription fmt(double rate, int ch);
// test hook: play a file into the input ring at real-time pace (stands in for a 48 kHz mic)
static void *feeder(void *arg) {
    CFURLRef u = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)g.feed_path, strlen(g.feed_path), false);
    ExtAudioFileRef f; CK(ExtAudioFileOpenURL(u, &f));
    AudioStreamBasicDescription cl = fmt(g.dev_in_rate, 1);
    CK(ExtAudioFileSetProperty(f, kExtAudioFileProperty_ClientDataFormat, sizeof cl, &cl));
    int chunk = (int)g.dev_in_rate / 100; float *b = calloc(chunk, 4); double t = now_ms();
    while (!atomic_load(&g.stop)) {
        UInt32 n = chunk; AudioBufferList bl = {1, {{1, n * 4, b}}}; CK(ExtAudioFileRead(f, &n, &bl));
        if (!n) break;
        ring_write(&g.in_ring, b, n); dispatch_semaphore_signal(g.sem);
        t += 10.0; double w = t - now_ms(); if (w > 0) usleep((useconds_t)(w * 1000));
    }
    ExtAudioFileDispose(f); atomic_store(&g.feed_done, 1); return NULL;
}
static AudioUnit make_unit(AudioDeviceID dev, int input) {
    AudioComponentDescription cd = {kAudioUnitType_Output, kAudioUnitSubType_HALOutput, kAudioUnitManufacturer_Apple, 0, 0};
    AudioUnit u; CK(AudioComponentInstanceNew(AudioComponentFindNext(NULL, &cd), &u));
    UInt32 on = 1, off = 0;
    CK(AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, input ? &on : &off, 4));
    CK(AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, input ? &off : &on, 4));
    CK(AudioUnitSetProperty(u, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &dev, sizeof dev));
    return u;
}
static AudioStreamBasicDescription fmt(double rate, int ch) {
    AudioStreamBasicDescription f = {0};
    f.mSampleRate = rate; f.mFormatID = kAudioFormatLinearPCM; f.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    f.mChannelsPerFrame = ch; f.mBitsPerChannel = 32; f.mFramesPerPacket = 1; f.mBytesPerFrame = f.mBytesPerPacket = 4 * ch;
    return f;
}
static void on_sigint(int s) { atomic_store(&g.stop, 1); }
static int cmp_d(const void *a, const void *b) { double x = *(double *)a, y = *(double *)b; return (x > y) - (x < y); }

static int cmd_live(int argc, char **argv) {
    const char *in_name = arg_s(argc, argv, "--input", NULL), *out_name = arg_s(argc, argv, "--output", "BlackHole 2ch");
    g.rate = arg_i(argc, argv, "--rate", 16000); int layers = arg_i(argc, argv, "--layers", 4), secs = arg_i(argc, argv, "--seconds", 0);
    g.feed_path = arg_s(argc, argv, "--input-file", NULL);
    AudioDeviceID din = g.feed_path ? 0 : find_dev(in_name, 1), dout = find_dev(out_name, 0);
    if ((!din && !g.feed_path) || !dout) { fprintf(stderr, "device not found (input %s, output %s). Run: reuse devices\n", in_name ? in_name : "default", out_name); return 1; }
    char ni[256], no[256]; if (din) dev_name(din, ni, sizeof ni); else snprintf(ni, sizeof ni, "file %s", g.feed_path); dev_name(dout, no, sizeof no);
    g.live = re_live_create(weights_path(), g.rate, layers);
    if (g.live) re_live_set_threads(g.live, arg_i(argc, argv, "--threads", auto_threads(arg_i(argc, argv, "--rate", 16000))));
    g.realtime = arg_i(argc, argv, "--realtime", 1);
    if (g.realtime) re_set_realtime(1000.0 * re_live_hop(g.live) / g.rate, 0.75 * 1000.0 * re_live_hop(g.live) / g.rate);
    if (!g.live) { fprintf(stderr, "cannot load model (weights %s, rate %d, layers %d)\n", weights_path(), g.rate, layers); return 1; }
    g.hop = re_live_hop(g.live); g.dev_in_rate = din ? dev_rate(din) : arg_i(argc, argv, "--feed-rate", 48000);
    const char *tap = arg_s(argc, argv, "--tap", NULL); if (tap) { g.tap_cap = (long)g.rate * 600; g.tap = malloc(g.tap_cap * 4); } g.out_ch = 2;
    g.proc_cap = 1 << 20; g.proc_ms = malloc(g.proc_cap * sizeof(double));
    ring_init(&g.in_ring, (int)g.dev_in_rate * 2); ring_init(&g.out_ring, g.rate * 2); g.sem = dispatch_semaphore_create(0);
    g.render_buf = calloc(16384, 4);
    if ((int)g.dev_in_rate != g.rate) {
        AudioStreamBasicDescription a = fmt(g.dev_in_rate, 1), b = fmt(g.rate, 1);
        CK(AudioConverterNew(&a, &b, &g.conv));
        UInt32 q = kAudioConverterQuality_High; AudioConverterSetProperty(g.conv, kAudioConverterSampleRateConverterQuality, 4, &q);
    }
    // input unit: client format = mono float at the device rate (AUHAL does not resample input)
    if (din) {
        g.in_unit = make_unit(din, 1);
        AudioStreamBasicDescription fi = fmt(g.dev_in_rate, 1);
        CK(AudioUnitSetProperty(g.in_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &fi, sizeof fi));
        AURenderCallbackStruct icb = {in_cb, NULL};
        CK(AudioUnitSetProperty(g.in_unit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0, &icb, sizeof icb));
    }
    // output unit: client format = 2ch interleaved float at the model rate (AUHAL resamples output)
    g.out_unit = make_unit(dout, 0);
    AudioStreamBasicDescription fo = fmt(g.rate, g.out_ch);
    CK(AudioUnitSetProperty(g.out_unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &fo, sizeof fo));
    AURenderCallbackStruct ocb = {out_cb, NULL};
    CK(AudioUnitSetProperty(g.out_unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &ocb, sizeof ocb));
    if (din) CK(AudioUnitInitialize(g.in_unit)); CK(AudioUnitInitialize(g.out_unit));
    int cushion = arg_i(argc, argv, "--cushion", 3);  // hops of jitter buffer before output starts
    float pre[4096] = {0}; ring_write(&g.out_ring, pre, g.hop * cushion);
    pthread_t th; pthread_create(&th, NULL, worker, NULL);
    signal(SIGINT, on_sigint);
    if (din) CK(AudioOutputUnitStart(g.in_unit)); CK(AudioOutputUnitStart(g.out_unit));
    pthread_t fth; if (g.feed_path) pthread_create(&fth, NULL, feeder, NULL);
    if (g.realtime && arg_i(argc, argv, "--workgroup", 1)) {  // join the output device's IO workgroup (Apple's pattern for helper RT threads)
        os_workgroup_t wg = NULL; UInt32 wsz = sizeof wg;
        if (!AudioUnitGetProperty(g.out_unit, kAudioOutputUnitProperty_OSWorkgroup, kAudioUnitScope_Global, 0, &wg, &wsz) && wg) { re_set_workgroup(wg); atomic_store(&g.wg, wg); }
        else fprintf(stderr, "note: output device has no IO workgroup\n");
    }
    printf("reuse live: %s (%.0f Hz) -> RE-USE %d Hz, %d layers -> %s | latency ~%.0f ms | Enter = toggle Enhanced/Raw, Ctrl+C = stop\n",
           ni, g.dev_in_rate, g.rate, layers, no, 1000.0 * g.hop * (2 + cushion) / g.rate);
    int enhanced = 1; double t_end = secs ? now_ms() + secs * 1000.0 : 0;
    if (!secs) {
        fd_set fs; struct timeval tv;
        while (!atomic_load(&g.stop)) {
            FD_ZERO(&fs); FD_SET(0, &fs); tv.tv_sec = 0; tv.tv_usec = 200000;
            if (select(1, &fs, NULL, NULL, &tv) > 0) { char line[64]; if (!fgets(line, sizeof line, stdin)) break;
                enhanced = !enhanced; re_live_set_enhanced(g.live, enhanced); printf(enhanced ? "ENHANCED\n" : "RAW\n"); fflush(stdout); }
        }
    } else while (!atomic_load(&g.stop) && now_ms() < t_end && !(g.feed_path && atomic_load(&g.feed_done))) usleep(50000);
    if (g.feed_path) usleep(300000);  // let the tail drain
    atomic_store(&g.stop, 1);
    if (din) AudioOutputUnitStop(g.in_unit); AudioOutputUnitStop(g.out_unit); pthread_join(th, NULL);
    if (g.feed_path) pthread_join(fth, NULL);
    if (tap) {
        CFURLRef tu = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)tap, strlen(tap), false);
        AudioStreamBasicDescription tf = fmt(g.rate, 1); ExtAudioFileRef tfh;
        CK(ExtAudioFileCreateWithURL(tu, kAudioFileWAVEType, &tf, NULL, kAudioFileFlags_EraseFile, &tfh));
        AudioBufferList bl = {1, {{1, (UInt32)(g.tap_n * 4), g.tap}}}; CK(ExtAudioFileWrite(tfh, (UInt32)g.tap_n, &bl)); ExtAudioFileDispose(tfh);
        printf("tap: %.1f s written to %s\n", g.tap_n / (double)g.rate, tap);
    }
    long n = g.hops < g.proc_cap ? g.hops : g.proc_cap; double med = 0, p95 = 0;
    if (n) { qsort(g.proc_ms, n, sizeof(double), cmp_d); med = g.proc_ms[n / 2]; p95 = g.proc_ms[(long)(n * 0.95)]; }
    if (g.late) { printf("late hops at index:"); for (long i = 0; i < g.late && i < 16; i++) printf(" %ld", g.late_at[i]); printf("\n"); }
    printf("hops %ld, compute median %.2f ms, p95 %.2f ms (budget %.0f ms), late %ld, output underruns %d, input overruns %d\n",
           g.hops, med, p95, 1000.0 * g.hop / g.rate, g.late, atomic_load(&g.underruns), atomic_load(&g.overruns));
    return 0;
}

// ---------- file ----------
static int cmd_file(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: reuse file IN OUT [--rate 16000] [--layers 4]\n"); return 1; }
    int rate = arg_i(argc, argv, "--rate", 16000), layers = arg_i(argc, argv, "--layers", 4);
    void *L = re_live_create(weights_path(), rate, layers); if (!L) { fprintf(stderr, "cannot load model\n"); return 1; }
    re_live_set_threads(L, arg_i(argc, argv, "--threads", auto_threads(arg_i(argc, argv, "--rate", 16000))));
    int H = re_live_hop(L);
    CFURLRef iu = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)argv[2], strlen(argv[2]), false);
    CFURLRef ou = CFURLCreateFromFileSystemRepresentation(NULL, (const UInt8 *)argv[3], strlen(argv[3]), false);
    ExtAudioFileRef in, out; CK(ExtAudioFileOpenURL(iu, &in));
    AudioStreamBasicDescription src; UInt32 sz = sizeof src;
    CK(ExtAudioFileGetProperty(in, kExtAudioFileProperty_FileDataFormat, &sz, &src));
    AudioStreamBasicDescription cl = fmt(rate, 1);
    CK(ExtAudioFileSetProperty(in, kExtAudioFileProperty_ClientDataFormat, sizeof cl, &cl));  // decode + downmix + resample
    AudioStreamBasicDescription of = fmt(src.mSampleRate, 1);
    CK(ExtAudioFileCreateWithURL(ou, kAudioFileWAVEType, &of, NULL, kAudioFileFlags_EraseFile, &out));
    CK(ExtAudioFileSetProperty(out, kExtAudioFileProperty_ClientDataFormat, sizeof cl, &cl));  // write back at source rate
    float *x = calloc(H, 4), *y = calloc(H, 4); long hops = 0; int first = 1, eof = 0, pad_hops = 1;
    double t0 = now_ms();
    while (pad_hops > 0) {
        UInt32 got = 0;
        if (!eof) { while (got < (UInt32)H) { UInt32 n = H - got; AudioBufferList bl = {1, {{1, n * 4, x + got}}}; CK(ExtAudioFileRead(in, &n, &bl)); if (!n) { eof = 1; break; } got += n; } }
        if (got < (UInt32)H) memset(x + got, 0, (H - got) * 4);
        if (eof && got == 0) pad_hops--;  // one extra hop flushes the one-hop delay
        re_live_process(L, x, y); hops++;
        if (first) { first = 0; continue; }  // drop the first hop: output lags input by one hop
        AudioBufferList bl = {1, {{1, (UInt32)H * 4, y}}}; CK(ExtAudioFileWrite(out, H, &bl));
    }
    double dt = now_ms() - t0;
    ExtAudioFileDispose(in); ExtAudioFileDispose(out);
    printf("%s -> %s: %.1f s audio in %.2f s (%.0fx realtime), %d Hz model, %d layers\n", argv[2], argv[3], hops * (double)H / rate, dt / 1000, hops * (double)H / rate / (dt / 1000), rate, layers);
    return 0;
}

static int cmd_bench(int argc, char **argv) {
    int rate = arg_i(argc, argv, "--rate", 16000), layers = arg_i(argc, argv, "--layers", 4);
    if (arg_i(argc, argv, "--realtime", 1)) { re_set_realtime(20.0, 15.0); re_rt_self(20.0, 15.0); }  // same class as live (20 ms hop)
    void *L = re_live_create(weights_path(), rate, layers); if (!L) { fprintf(stderr, "cannot load model\n"); return 1; }
    re_live_set_threads(L, arg_i(argc, argv, "--threads", auto_threads(arg_i(argc, argv, "--rate", 16000))));
    int H = re_live_hop(L); float *x = calloc(H, 4), *y = calloc(H, 4); double t[500];
    // hops are paced every hop_ms like live audio: a real-time thread that runs back to back blows its budget and gets demoted
    mach_timebase_info_data_t tb; mach_timebase_info(&tb); uint64_t hop_ticks = (uint64_t)(1e6 * 1000.0 * H / rate * tb.denom / tb.numer), next = mach_absolute_time();
    struct rusage ru0, ru1; double w0 = 0;
    for (int i = 0; i < 520; i++) {
        if (i == 20) { getrusage(RUSAGE_SELF, &ru0); w0 = now_ms(); }
        for (int n = 0; n < H; n++) x[n] = 0.05f * sinf(0.01f * (i * H + n)) + 0.01f * ((rand() & 1023) / 512.f - 1);
        next += hop_ticks; double t0 = now_ms(); re_live_process(L, x, y); if (i >= 20) t[i - 20] = now_ms() - t0;
        if (mach_absolute_time() < next) mach_wait_until(next); else next = mach_absolute_time();
    }
    qsort(t, 500, sizeof(double), cmp_d);
    getrusage(RUSAGE_SELF, &ru1);
    double cpu = (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec + ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) * 1e3 + (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec + ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec) / 1e3;
    printf("%d Hz, %d layers: hop %.0f ms, compute median %.2f ms, p95 %.2f ms, max %.2f ms, busy %.0f%% of each hop, CPU %.2f cores\n", rate, layers, 1000.0 * H / rate, t[250], t[475], t[499], 100.0 * t[250] / (1000.0 * H / rate), cpu / (now_ms() - w0));
    return 0;
}

static int known(const char *a) {
    const char *ok[] = {"--input", "--output", "--rate", "--layers", "--seconds", "--input-file", "--tap", "--feed-rate", "--threads", "--cushion", "--realtime", "--workgroup", NULL};
    for (int i = 0; ok[i]; i++) if (!strcmp(a, ok[i])) return 1;
    return 0;
}
int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "";
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) cmd = "";
    if (!strcmp(cmd, "live"))  // refuse unknown flags: a typo must never silently open the default mic
        for (int i = 2; i < argc; i += 2) if (!known(argv[i])) { fprintf(stderr, "unknown option %s\n", argv[i]); return 2; }
    if (!strcmp(cmd, "devices")) return cmd_devices();
    if (!strcmp(cmd, "live")) return cmd_live(argc, argv);
    if (!strcmp(cmd, "file")) return cmd_file(argc, argv);
    if (!strcmp(cmd, "bench")) return cmd_bench(argc, argv);
    fprintf(stderr, "usage:\n  reuse devices\n  reuse live [--input NAME] [--output \"BlackHole 2ch\"] [--rate 16000] [--layers 4] [--threads 1..4 (default by rate)] [--cushion 3] [--realtime 1|0] [--seconds N]\n"
                    "  reuse file IN OUT [--rate 16000] [--layers 4]\n  reuse bench [--rate 16000] [--layers 4]\n");
    return 2;
}
