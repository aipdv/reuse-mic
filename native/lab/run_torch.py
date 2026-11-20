"""NVIDIA Real-time RE-USE in PyTorch (reuse_core.LiveEnhancer, NVIDIA's streaming code + CPU scan shim), hop by hop.
python run_torch.py IN.wav OUT.wav EXIT_LAYER   -> prints per-hop compute time (20 ms hops at the file's rate)."""
import sys, time; from pathlib import Path
sys.path.insert(0, str(Path.home() / "Coding/reuse-mic"))
import numpy as np, soundfile as sf, torch
from reuse_core import LiveEnhancer, load_model
x, sr = sf.read(sys.argv[1], dtype="float32", always_2d=True); x = x[:, 0]
model, cfg = load_model(); le = LiveEnhancer(model, cfg, sr, exit_layer=int(sys.argv[3]))
H = le.hop; n = len(x) // H + 1; x = np.pad(x, (0, n * H - len(x))); ys, ts = [], []
for i in range(n):
    t0 = time.perf_counter(); ys.append(le.process(torch.from_numpy(x[i * H:(i + 1) * H])).numpy()); ts.append(time.perf_counter() - t0)
y = np.concatenate(ys)[H:]  # drop the one-hop delay
sf.write(sys.argv[2], y[: len(x)], sr)
ts = np.array(ts[20:]) * 1e3
print(f"{sr} Hz, {sys.argv[3]} layers, torch threads {torch.get_num_threads()}: median {np.median(ts):.1f} ms, p95 {np.percentile(ts, 95):.1f} ms per {1000 * H / sr:.0f} ms hop")
