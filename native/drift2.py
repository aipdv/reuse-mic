"""Per-loop deviation from fresh-start output (same input each loop), with and without periodic state reset."""
import ctypes, sys; from pathlib import Path
import numpy as np, soundfile as sf
h = Path(__file__).parent; lib = ctypes.CDLL(str(h / "libreuse_engine.dylib"))
lib.re_live_create.restype = ctypes.c_void_p; lib.re_live_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
P = ctypes.POINTER(ctypes.c_float); lib.re_live_process.argtypes = [ctypes.c_void_p, P, P]; lib.re_live_reset.argtypes = [ctypes.c_void_p]
x, _ = sf.read("work/tiled_speech.wav", dtype="float32"); H = 320; loop = int(open("work/loop_len.txt").read())
secs = int(sys.argv[1]); x = x[:secs * 16000]
def run(reset_every_s):
    L = lib.re_live_create(str(h / "reuse_rt.bin").encode(), 16000, 4); out = []
    for i in range(len(x) // H):
        if reset_every_s and i and i % (reset_every_s * 50) == 0: lib.re_live_reset(L)
        s = np.ascontiguousarray(x[i*H:(i+1)*H]); y = np.zeros(H, np.float32)
        lib.re_live_process(L, s.ctypes.data_as(P), y.ctypes.data_as(P)); out.append(y)
    return np.concatenate(out)
for r in (0, 20):
    y = run(r); ref = y[:loop]
    stats = []
    for k in range(len(y) // loop):
        seg = y[k*loop:(k+1)*loop]; skip = 16000  # ignore first second of each loop
        snr = 10 * np.log10((ref[skip:]**2).sum() / ((seg[skip:] - ref[skip:])**2).sum())
        stats.append(f"{np.sqrt((seg**2).mean()):.4f}/{snr:4.1f}dB")
    print(f"reset every {r or 'never'} s: rms/SNR-vs-fresh per loop:", " ".join(stats))
