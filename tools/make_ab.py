"""In-place A/B/C: one video, audio switches Original -> RE-USE -> Inworld.

  .venv/bin/python make_ab.py input.mp4 work/orig.wav work/reuse.wav work/inworld.wav out.mp4

A white bar sweeps left to right at each switch; a pill top-right names the active model.
Levels are left as each model outputs them (no loudness matching).
"""
import subprocess
import sys

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

video, *wavs, out = sys.argv[1:]
LABELS = __import__("os").environ.get("LABELS", "Original|NVIDIA RE-USE|Inworld Voice Isolator").split("|")
SWEEP, FADE = 0.6, 0.03  # seconds: bar sweep, audio crossfade at each switch

tracks = [sf.read(w, dtype="float32", always_2d=True)[0][:, 0] for w in wavs]
sr = sf.info(wavs[0]).samplerate
n = min(min(len(t) for t in tracks), 26 * sr)  # 6 + 10 + 10 s
cuts = [0, 6 * sr, 16 * sr, n]  # 6 s original, then 10 s each
f = int(FADE * sr)
gain = np.zeros((3, n), dtype=np.float32)
for i in range(3):
    a, b = cuts[i], cuts[i + 1]
    gain[i, a:b] = 1
    if i:  # linear crossfade centred on the cut
        ramp = np.linspace(0, 1, 2 * f, dtype=np.float32)
        gain[i, a - f:a + f] = ramp
        gain[i - 1, a - f:a + f] = 1 - ramp
mix = sum(g * t[:n] for g, t in zip(gain, tracks))
sf.write("work/ab.wav", mix, sr, subtype="FLOAT")

font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
for i, text in enumerate(LABELS):
    w = int(font.getlength(text)) + 40
    im = Image.new("RGBA", (w, 48), (0, 0, 0, 0))
    ImageDraw.Draw(im).rounded_rectangle((0, 0, w - 1, 47), 24, fill=(0, 0, 0, 170))
    ImageDraw.Draw(im).text((20, 9), text, font=font, fill=(255, 255, 255, 255))
    im.save(f"work/label{i}.png")
Image.new("RGBA", (8, 2000), (255, 255, 255, 230)).save("work/bar.png")

t = [c / sr for c in cuts]
fc = "[0:v]null[v0];"
for i in range(3):  # label i visible during its third
    fc += (f"[v{i}][{2 + i}:v]overlay=x=W-w-24:y=24:"
           f"enable='between(t,{t[i]:.3f},{t[i + 1]:.3f})'[v{i + 1}];")
bar_x = "+".join(f"if(between(t,{c - SWEEP / 2:.3f},{c + SWEEP / 2:.3f}),(t-{c - SWEEP / 2:.3f})/{SWEEP}*W,0)"
                 for c in t[1:3])
fc += f"[v3][5:v]overlay=x='{bar_x}-w':y=0:eval=frame,format=yuv420p[v]"
subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-y", "-i", video, "-i", "work/ab.wav",
                *sum((["-i", f"work/label{i}.png"] for i in range(3)), []), "-i", "work/bar.png",
                "-filter_complex", fc, "-map", "[v]", "-map", "1:a", "-shortest",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "2",
                "-c:a", "aac", "-b:a", "192k", "-t", f"{n / sr:.3f}", out], check=True)
print(out)
