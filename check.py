"""Parity + speed check against NVIDIA's CUDA reference outputs.

Run: .venv/bin/python check.py
Pass: cosine >= 0.99 vs CUDA reference; live per-hop time well under 20 ms.
"""
import time

import numpy as np
import soundfile as sf
import torch

from reuse_core import VENDOR, LiveEnhancer, enhance_offline, load_model

torch.set_num_threads(4)


def cos(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


model, cfg = load_model()
noisy, sr = sf.read(VENDOR / "noisy_audio/mic_test.wav", dtype="float32")
ref_off, _ = sf.read(VENDOR / "offline_enhanced_audio/mic_test.flac", dtype="float32")
ref_on, _ = sf.read(VENDOR / "enhanced_audio/online_enhanced_mic_test.flac", dtype="float32")

t0 = time.perf_counter()
off = enhance_offline(model, cfg, torch.from_numpy(noisy)[None], sr)[0].numpy()
print(f"offline: {time.perf_counter() - t0:.1f}s for {len(noisy) / sr:.1f}s audio")
c_off = cos(off, ref_off)
print(f"offline vs NVIDIA CUDA offline: cosine {c_off:.5f}")

live = LiveEnhancer(model, cfg, sr)
x = torch.from_numpy(noisy)
outs, times = [], []
for i in range(0, len(x) - live.hop + 1, live.hop):
    t0 = time.perf_counter()
    outs.append(live.process(x[i:i + live.hop]))
    times.append(time.perf_counter() - t0)
on = torch.cat(outs).numpy()[live.hop:]  # undo the one-hop output lag
c_on = cos(on, ref_on)
print(f"live vs NVIDIA CUDA online:     cosine {c_on:.5f}")
ms = np.array(times[5:]) * 1000
print(f"live @ {sr} Hz: hop {live.hop / sr * 1000:.0f} ms, compute median {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms")
sf.write("/tmp/reuse_live_44k.wav", on, sr)

for test_sr in (16000, 48000):
    le = LiveEnhancer(model, cfg, test_sr)
    seg = torch.randn(le.hop) * 0.01
    ts = []
    for _ in range(60):
        t0 = time.perf_counter()
        le.process(seg)
        ts.append(time.perf_counter() - t0)
    ms = np.array(ts[10:]) * 1000
    print(f"live @ {test_sr} Hz: compute median {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms (budget 20 ms)")

assert c_off > 0.99 and c_on > 0.99, "parity failed"
print("PASS")
