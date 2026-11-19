"""Offline enhancement with NVIDIA NemotronLabs RE-USE (30-layer, full-clip model) on Mac CPU.

  .venv/bin/python enhance_file.py in.wav out.wav [--chunk 3.0]

Mirrors NVIDIA's inference_chunk.py: overlapping chunks (50% hop), Hann-weighted overlap-add.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "NemotronLabs_REUSE"
sys.path.insert(0, str(ROOT / "shim"))
sys.path.insert(0, str(VENDOR))
from models.generator_SEMamba_time_d4 import SEMamba  # noqa: E402
from models.stfts import mag_phase_istft, mag_phase_stft  # noqa: E402
from utils.util import load_config  # noqa: E402


def make_even(v):
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


@torch.no_grad()
def enhance(model, cfg, wav, sr):  # wav: [1, N]
    base = cfg["stft_cfg"]["sampling_rate"]
    n_fft = make_even(cfg["stft_cfg"]["n_fft"] * sr // base)
    hop = make_even(cfg["stft_cfg"]["hop_size"] * sr // base)
    win = make_even(cfg["stft_cfg"]["win_size"] * sr // base)
    cf = cfg["model_cfg"]["compress_factor"]
    mag, pha, _ = mag_phase_stft(wav, n_fft=n_fft, hop_size=hop, win_size=win, compress_factor=cf)
    amp, ph, _ = model(mag, pha)
    m = torch.expm1(F.relu(amp))
    amp[:, :, ((m == 0).sum(1) / m.shape[1] > 0.5)[0]] = 0  # NVIDIA's sweep-artifact guard
    out = mag_phase_istft(amp, ph, n_fft, hop, win, cf)
    return F.pad(out, (0, max(0, wav.shape[1] - out.shape[1])))[:, : wav.shape[1]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=float, default=3.0, help="seconds; smaller = less RAM")
    a = ap.parse_args()

    torch.set_num_threads(4)
    cfg = load_config(next((VENDOR / "recipes").glob("*.yaml")))
    model = SEMamba.from_pretrained(str(VENDOR), cfg=cfg).eval()
    x, sr = sf.read(a.inp, dtype="float32", always_2d=True)
    wav = torch.from_numpy(x[:, 0].copy())[None]  # mono

    size = int(a.chunk * sr)
    hop = size // 2
    w = torch.clamp(torch.hann_window(size), min=1e-7)
    acc, wsum = torch.zeros_like(wav), torch.zeros_like(wav)
    n = max(1, math.ceil((wav.shape[1] - size) / hop) + 1)
    t0 = time.perf_counter()
    for i in range(n):
        s = min(i * hop, max(0, wav.shape[1] - size))  # last chunk aligns to the end
        seg = wav[:, s:s + size]
        y = enhance(model, cfg, seg, sr)
        acc[:, s:s + y.shape[1]] += y * w[: y.shape[1]]
        wsum[:, s:s + y.shape[1]] += w[: y.shape[1]]
        print(f"chunk {i + 1}/{n} done, {time.perf_counter() - t0:.0f}s", flush=True)
    sf.write(a.out, (acc / wsum)[0].numpy(), sr)
    print(f"wrote {a.out}: {wav.shape[1] / sr:.1f}s audio in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
