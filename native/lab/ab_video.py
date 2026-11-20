"""A/B video: per clip, Noisy -> RE-USE 12 kHz -> RE-USE 48 kHz; each segment shows the spectrogram of what plays.

  python ab_video.py OUT.mp4 "Clip name|noisy.wav|out12k.wav|out48k.wav" [...]
Needs ffmpeg. Audio is resampled to 48 kHz and normalised once for the whole video (levels stay comparable).
"""
import os, subprocess, sys, tempfile
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF
from PIL import Image, ImageDraw, ImageFont

out, clips = sys.argv[1], [c.split("|") for c in sys.argv[2:]]
SR, GAP, W, H = 48000, 0.6, 1280, 720
LABELS = ["Noisy input", "RE-USE 12 kHz (old)", "RE-USE 48 kHz (new)"]
COLORS = ["#888888", "#d9822b", "#2bb673"]
def font(n):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"):
        try: return ImageFont.truetype(p, n)
        except OSError: pass
    return ImageFont.load_default()
F_T, F_L, F_S = font(34), font(30), font(20)

def rd(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True); x = x[:, 0]
    return AF.resample(torch.from_numpy(x), sr, SR).numpy() if sr != SR else x

def spec(x, w, h):  # log spectrogram 0..24 kHz, 80 dB range, high freq on top
    N, HOP = 2048, 256
    fr = np.lib.stride_tricks.sliding_window_view(np.pad(x, (0, N)), N)[::HOP] * np.hanning(N)
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)).T + 1e-6)
    S = np.clip((S - (S.max() - 80)) / 80, 0, 1)[::-1]
    g = Image.fromarray((S * 255).astype(np.uint8)).resize((w, h))
    return Image.merge("RGB", (g.point(lambda v: min(255, v * 2)), g.point(lambda v: max(0, v * 2 - 200) if v > 100 else 0),
                               g.point(lambda v: 255 - abs(v * 2 - 255) // 2 if v < 128 else 0)))

tmp = tempfile.mkdtemp(); audio, lines = [], []
peak = max(np.abs(rd(p)).max() for c in clips for p in c[1:])
for ci, (name, *paths) in enumerate(clips):
    for k, p in enumerate(paths):
        x = rd(p) / peak * 0.9; audio += [x, np.zeros(int(GAP * SR), np.float32)]
        img = Image.new("RGB", (W, H), "#111111"); d = ImageDraw.Draw(img)
        d.text((40, 24), "RE-USE live mic on an M3 Air: 12 kHz vs 48 kHz", font=F_T, fill="white")
        d.text((40, 70), name, font=F_S, fill="#bbbbbb")
        for j, lab in enumerate(LABELS):  # progress pills, active one filled
            x0 = 40 + j * 405; box = [x0, 110, x0 + 390, 160]
            d.rounded_rectangle(box, 12, fill=COLORS[j] if j == k else None, outline=COLORS[j], width=3)
            d.text((x0 + 18, 120), lab, font=F_L, fill="black" if j == k else COLORS[j])
        sy, sh = 190, 500; img.paste(spec(x, W - 120, sh), (100, sy))
        for khz in (4, 6, 8, 12, 16, 20):
            y = int(sy + sh * (1 - khz / 24)); d.line([(88, y), (100, y)], fill="white"); d.text((20, y - 11), f"{khz} kHz", font=F_S, fill="white")
        y6 = int(sy + sh * (1 - 6 / 24)); d.line([(100, y6), (W - 20, y6)], fill="#d9822b", width=1)
        d.text((W - 270, y6 - 26), "12 kHz model stops here", font=F_S, fill="#d9822b")
        f = os.path.join(tmp, f"f{ci}_{k}.png"); img.save(f)
        lines += [f"file '{f}'", f"duration {len(x) / SR + GAP:.3f}"]
lines.append(f"file '{f}'")  # concat demuxer: last frame needs a trailing entry
open(os.path.join(tmp, "list.txt"), "w").write("\n".join(lines) + "\n")
sf.write(os.path.join(tmp, "a.wav"), np.concatenate(audio), SR, subtype="PCM_16")
subprocess.run(["nice", "-n", "10", "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(tmp, "list.txt"),
                "-i", os.path.join(tmp, "a.wav"), "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "2",
                "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
print(out, f"{sum(len(a) for a in audio) / SR:.1f} s")
