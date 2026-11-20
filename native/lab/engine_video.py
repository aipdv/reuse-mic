"""Head-to-head video: the source video plays while its audio is replaced by engine A, then engine B.
Below: original spectrogram (top) and the playing engine's spectrogram (bottom) with a moving playhead.
Right: one card per engine (layers, rate, time per 20 ms hop); the playing one is highlighted.

  python engine_video.py OUT.mp4 VIDEO ORIG.wav "title|line1|line2|line3|out.wav" "title|...|out.wav"
Needs ffmpeg. 1920x1080, 24 fps.
"""
import os, subprocess, sys, tempfile
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFont

out, video, orig, *engines = sys.argv[1:]
engines = [e.split("|") for e in engines]
W, H, FPS = 1920, 1080, 24
VX, VY, VW, VH = 24, 24, 1024, 576           # video panel
SX, SW, SH, SY0, SY1 = 24, 1872, 214, 628, 858  # spectrogram panels (original, processed)
COL = ["#d9822b", "#2bb673"]
def font(n):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"):
        try: return ImageFont.truetype(p, n)
        except OSError: pass
    return ImageFont.load_default()
F1, F2, F3, F4 = font(40), font(30), font(24), font(20)

def spec(path, w, h, dur, ref=None):
    x, sr = sf.read(path, dtype="float32", always_2d=True); x = np.pad(x[:, 0], (0, max(0, int(dur * sr) - len(x))))[: int(dur * sr)]
    N, HOP = 2048, 256
    fr = np.lib.stride_tricks.sliding_window_view(np.pad(x, (0, N)), N)[::HOP][: len(x) // HOP] * np.hanning(N)
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)).T + 1e-6)
    ref = S.max() if ref is None else ref    # one reference (the original's peak) for every panel: same colours = same level
    S = np.clip((S - (ref - 80)) / 80, 0, 1)[::-1]
    g = Image.fromarray((S * 255).astype(np.uint8)).resize((w, h))
    return ref, Image.merge("RGB", (g.point(lambda v: min(255, v * 2)), g.point(lambda v: max(0, v * 2 - 200) if v > 100 else 0),
                               g.point(lambda v: 255 - abs(v * 2 - 255) // 2 if v < 128 else 0))), sr

dur = sf.info(orig).duration
tmp = tempfile.mkdtemp(); parts = []
REF, sp_orig, sr0 = spec(orig, SW, SH, dur)
for k, (title, *lines, wav) in enumerate(engines):
    img = Image.new("RGB", (W, H), "#101010"); d = ImageDraw.Draw(img)
    d.rectangle([VX - 2, VY - 2, VX + VW + 1, VY + VH + 1], outline="#333333")
    cx = VX + VW + 36
    d.text((cx, 26), "Same NVIDIA RE-USE weights, two engines", font=F2, fill="white")
    d.text((cx, 66), "M3 MacBook Air, 8 GB, CPU. Audio = the engine marked PLAYING.", font=F4, fill="#aaaaaa")
    for j, (t2, *l2, _) in enumerate(engines):  # cards
        y0 = 106 + j * 248; on = j == k
        d.rounded_rectangle([cx, y0, W - 24, y0 + 236], 16, fill="#1d1d1d" if on else "#141414", outline=COL[j] if on else "#333333", width=4 if on else 2)
        d.text((cx + 24, y0 + 14), ("PLAYING  " if on else "") + t2, font=F2, fill=COL[j] if on else "#777777")
        for i, ln in enumerate(l2): d.text((cx + 24, y0 + 56 + (0 if not i else 18 + i * 38)), ln, font=F3 if i else F1, fill="white" if on else "#666666")
    _r, sp_proc, _ = spec(wav, SW, SH, dur, REF)
    for y, im, lab, c in ((SY0, sp_orig, "Original audio", "#cccccc"), (SY1, sp_proc, title, COL[k])):
        img.paste(im, (SX, y)); d.rectangle([SX, y, SX + SW - 1, y + SH - 1], outline="#333333")
        d.rounded_rectangle([SX + 10, y + 8, SX + 22 + 13 * len(lab), y + 40], 8, fill="#000000"); d.text((SX + 18, y + 12), lab, font=F4, fill=c)
        nyq = 22050 if im is sp_orig else _ / 2
        for khz in (5, 10, 15, 20):
            yy = int(y + SH * (1 - khz * 1000 / (sr0 / 2 if im is sp_orig else nyq)))
            if y < yy < y + SH: d.line([(SX + SW - 70, yy), (SX + SW - 56, yy)], fill="white"); d.text((SX + SW - 52, yy - 10), f"{khz}k", font=F4, fill="white")
    bg = os.path.join(tmp, f"bg{k}.png"); img.save(bg)
    seg = os.path.join(tmp, f"seg{k}.mp4")
    fc = (f"[1:v]scale={VW}:{VH},setsar=1,fps={FPS}[v];[0:v][v]overlay={VX}:{VY}[a];"
          f"[a][2:v]overlay=x='{SX}+{SW}*t/{dur:.3f}':y={SY0}:eval=frame[o]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(FPS), "-i", bg, "-i", video,
                    "-f", "lavfi", "-i", f"color=c=white:s=3x{SY1 + SH - SY0}:r={FPS}", "-i", wav,
                    "-filter_complex", fc, "-map", "[o]", "-map", "3:a", "-t", f"{dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-r", str(FPS), "-threads", "2", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", seg], check=True)
    parts.append(seg)
open(os.path.join(tmp, "list.txt"), "w").write("".join(f"file '{p}'\n" for p in parts))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", os.path.join(tmp, "list.txt"), "-c", "copy", out], check=True)
print(out)
