"""3-part A/B video: Standard -> macOS Voice Isolation -> NVIDIA RE-USE.

  .venv/bin/python make_video.py [--speed 1.25]   (speed keeps pitch: ffmpeg atempo)
Each part gets a title card plus a live waveform. Loudness is matched (-18 LUFS) so the
comparison is about quality, not volume.
"""
import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

D = Path(__file__).resolve().parent / "demo"
ap = argparse.ArgumentParser()
ap.add_argument("--speed", type=float, default=1.0)
SPEED = ap.parse_args().speed
OUT = Path.home() / "Desktop" / ("mic-modes-comparison.mp4" if SPEED == 1 else f"mic-modes-comparison-{SPEED:g}x.mp4")
W, H = 1280, 720
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
REG = "/System/Library/Fonts/Supplemental/Arial.ttf"

PARTS = [
    ("1", "Standard", "External mic, no processing", D / "standard.wav", "0x9CA3AF"),
    ("2", "macOS Voice Isolation", "Apple's built-in Mic Mode (separate take)", D / "vi.wav", "0x60A5FA"),
    ("3", "NVIDIA RE-USE", "NemotronLabs RE-USE on take 1, 48 kHz", D / "spaces" / "nemotron_48k.wav", "0xF59E0B"),
]


def card(num, title, sub, path):
    im = Image.new("RGB", (W, H), (14, 14, 18))
    d = ImageDraw.Draw(im)
    d.text((80, 120), f"Part {num} of 3", font=ImageFont.truetype(REG, 34), fill=(150, 150, 160))
    d.text((80, 175), title, font=ImageFont.truetype(BOLD, 76), fill=(245, 245, 245))
    d.text((80, 285), sub, font=ImageFont.truetype(REG, 36), fill=(185, 185, 195))
    im.save(path)


segs = []
for num, title, sub, audio, color in PARTS:
    png, mp4 = D / f"card{num}.png", D / f"seg{num}_{SPEED:g}x.mp4"
    card(num, title, sub, png)
    fc = (f"[1:a]aresample=48000,loudnorm=I=-18:TP=-1.5:LRA=11,atempo={SPEED},aresample=48000,asplit[a][w];"
          f"[w]showwaves=s=1120x200:mode=cline:rate=30:colors={color}[wv];"
          f"[0:v][wv]overlay=80:430,format=yuv420p[v]")
    subprocess.run(["nice", "-n", "10", "ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "30",
                    "-i", str(png), "-i", str(audio), "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                    "-shortest", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "1", str(mp4)], check=True)
    segs.append(mp4)

lst = D / "concat.txt"
lst.write_text("".join(f"file '{s}'\n" for s in segs))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                "-c", "copy", str(OUT)], check=True)
print(OUT)
