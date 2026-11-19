"""Does output loudness drift over a long stream? native vs torch LiveEnhancer on the same looped input."""
import ctypes, sys; from pathlib import Path
sys.path.insert(0, str(Path.home() / "Coding/reuse-mic"))
import numpy as np, soundfile as sf, torch
from reuse_core import LiveEnhancer, load_model
secs = int(sys.argv[1]) if len(sys.argv) > 1 else 180
h = Path(__file__).parent; lib = ctypes.CDLL(str(h / "libreuse_engine.dylib"))
lib.re_live_create.restype = ctypes.c_void_p; lib.re_live_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
P = ctypes.POINTER(ctypes.c_float); lib.re_live_process.argtypes = [ctypes.c_void_p, P, P]
L = lib.re_live_create(str(h / "reuse_rt.bin").encode(), 16000, 4); H = 320
torch.set_num_threads(2); model, cfg = load_model(); le = LiveEnhancer(model, cfg, 16000, exit_layer=4)
x, _ = sf.read("work/long10min.wav", dtype="float32"); x = x[:secs * 16000]
a, b = [], []
for i in range(len(x) // H):
    s = np.ascontiguousarray(x[i*H:(i+1)*H]); y = np.zeros(H, np.float32)
    lib.re_live_process(L, s.ctypes.data_as(P), y.ctypes.data_as(P)); a.append(y)
    b.append(le.process(torch.from_numpy(s)).numpy())
a, b = np.concatenate(a), np.concatenate(b); seg = 16000 * 26  # one loop of the input
ra = [np.sqrt((a[i:i+seg]**2).mean()) for i in range(0, len(a) - seg + 1, seg)]
rb = [np.sqrt((b[i:i+seg]**2).mean()) for i in range(0, len(b) - seg + 1, seg)]
print("rms per 26 s loop, native:", " ".join(f"{v:.4f}" for v in ra))
print("rms per 26 s loop, torch :", " ".join(f"{v:.4f}" for v in rb))
print(f"native vs torch SNR over {secs}s: {10*np.log10((b**2).sum()/((a-b)**2).sum()):.1f} dB")
