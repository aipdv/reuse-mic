"""Frame-by-frame parity: native engine vs PyTorch StreamingSEMamba (look-ahead 0)."""
import ctypes, sys; from pathlib import Path
sys.path.insert(0, str(Path.home() / "Coding/reuse-mic"))
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF
from reuse_core import LiveEnhancer, load_model
rate = int(sys.argv[1]) if len(sys.argv) > 1 else 12000; layers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
N = int(sys.argv[3]) if len(sys.argv) > 3 else 150
here = Path(__file__).parent
lib = ctypes.CDLL(str(here / "libreuse_engine.dylib"))
lib.re_create.restype = ctypes.c_void_p; lib.re_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
P = ctypes.POINTER(ctypes.c_float)
lib.re_step.argtypes = [ctypes.c_void_p, P, P, P, P]; lib.re_reset.argtypes = [ctypes.c_void_p]
torch.set_num_threads(2)
model, cfg = load_model(); le = LiveEnhancer(model, cfg, rate, exit_layer=layers)
nb = le.n_fft // 2 + 1
eng = lib.re_create(str(here / "reuse_rt.bin").encode(), nb, layers); assert eng
wav, sr0 = sf.read(Path.home() / "Coding/reuse-mic/vendor/Realtime_REUSE/noisy_audio/mic_test.wav", dtype="float32")
wav = AF.resample(torch.from_numpy(wav), sr0, rate)
buf = torch.zeros(le.hop); errs = []; worst = 0
for i in range(N):
    x = wav[i * le.hop:(i + 1) * le.hop]
    spec = torch.fft.rfft(torch.cat([buf, x]) * le.w); buf = x
    mag = torch.log1p(spec.abs()).float(); pha = torch.angle(spec).float()
    with torch.no_grad():
        mt, pt = le.sm(mag[None, :, None], pha[None, :, None], layers, 0)
    mt, pt = mt[0].numpy(), pt[0].numpy()
    mo = np.zeros(nb, np.float32); po = np.zeros(nb, np.float32)
    m_np, p_np = mag.numpy().copy(), pha.numpy().copy()
    lib.re_step(eng, m_np.ctypes.data_as(P), p_np.ctypes.data_as(P), mo.ctypes.data_as(P), po.ctypes.data_as(P))
    em = np.abs(mo - mt).max(); dp = np.angle(np.exp(1j * (po - pt)))  # wrap phase
    w = np.expm1(np.maximum(mt, 0))  # weight phase error by magnitude
    ep = float(np.sqrt((w * dp ** 2).sum() / (w.sum() + 1e-9)))
    errs.append((em, ep)); worst = max(worst, em)
e = np.array(errs)
print(f"rate {rate} layers {layers} frames {N}: mag max|d| median {np.median(e[:,0]):.2e} worst {e[:,0].max():.2e}; "
      f"phase (mag-weighted rms rad) median {np.median(e[:,1]):.2e} worst {e[:,1].max():.2e}")
