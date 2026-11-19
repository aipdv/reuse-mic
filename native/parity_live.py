"""Waveform parity: native re_live_process vs reuse_core.LiveEnhancer.process (look-ahead 0)."""
import ctypes, sys; from pathlib import Path
sys.path.insert(0, str(Path.home() / "Coding/reuse-mic"))
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF
from reuse_core import LiveEnhancer, load_model
rate, layers = int(sys.argv[1]), int(sys.argv[2])
h = Path(__file__).parent; lib = ctypes.CDLL(str(h / "libreuse_engine.dylib"))
lib.re_live_create.restype = ctypes.c_void_p; lib.re_live_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
P = ctypes.POINTER(ctypes.c_float); lib.re_live_process.argtypes = [ctypes.c_void_p, P, P]
lib.re_live_hop.argtypes = [ctypes.c_void_p]
L = lib.re_live_create(str(h / "reuse_rt.bin").encode(), rate, layers); assert L
hop = lib.re_live_hop(L)
torch.set_num_threads(2); model, cfg = load_model(); le = LiveEnhancer(model, cfg, rate, exit_layer=layers)
assert le.hop == hop, (le.hop, hop)
wav, sr0 = sf.read(Path.home() / "Coding/reuse-mic/vendor/Realtime_REUSE/noisy_audio/mic_test.wav", dtype="float32")
wav = AF.resample(torch.from_numpy(wav), sr0, rate).numpy(); n = len(wav) // hop
a, b = [], []
for i in range(n):
    x = np.ascontiguousarray(wav[i * hop:(i + 1) * hop]); y = np.zeros(hop, np.float32)
    lib.re_live_process(L, x.ctypes.data_as(P), y.ctypes.data_as(P)); a.append(y)
    b.append(le.process(torch.from_numpy(x)).numpy())
a, b = np.concatenate(a), np.concatenate(b)
snr = 10 * np.log10((b ** 2).sum() / ((a - b) ** 2).sum())
print(f"rate {rate} layers {layers}: {n} hops, waveform SNR native vs torch {snr:.1f} dB, max|d| {np.abs(a-b).max():.2e}, rms {np.sqrt((b**2).mean()):.3f}")
sf.write(h / f"native_{rate}_{layers}.wav", a, rate); sf.write(h / f"torch_{rate}_{layers}.wav", b, rate)
