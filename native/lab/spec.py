"""Stacked spectrograms (0 to 22 kHz) of several wavs: python spec.py OUT.png LABEL=file.wav ... (first 12 s)."""
import sys, numpy as np, soundfile as sf
from PIL import Image, ImageDraw
out, items = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
N, HOP, W, H = 2048, 256, 1200, 300
rows = []
for label, path in items:
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = x[: 12 * sr, 0]
    fr = np.lib.stride_tricks.sliding_window_view(x, N)[::HOP] * np.hanning(N)
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)).T + 1e-6)          # [bins, frames]
    S = np.clip((S - (S.max() - 80)) / 80, 0, 1)[::-1]                   # 80 dB range, high freq on top
    img = Image.fromarray((S * 255).astype(np.uint8)).resize((W, H))
    rgb = Image.merge("RGB", (img.point(lambda v: min(255, v * 2)), img.point(lambda v: max(0, v * 2 - 200) if v > 100 else 0), img.point(lambda v: 255 - abs(v * 2 - 255) // 2 if v < 128 else 0)))
    d = ImageDraw.Draw(rgb); ny = sr / 2
    for khz in (6, 8, 12, 16, 20):
        y = int(H * (1 - khz * 1000 / ny)); d.line([(0, y), (14, y)], fill="white"); d.text((18, y - 6), f"{khz} kHz", fill="white")
    d.rectangle([W - 330, 4, W - 4, 22], fill="black"); d.text((W - 324, 8), label, fill="white")
    rows.append(rgb)
canvas = Image.new("RGB", (W, H * len(rows) + 4 * (len(rows) - 1)), "black")
for i, r in enumerate(rows): canvas.paste(r, (0, i * (H + 4)))
canvas.save(out); print(out)
