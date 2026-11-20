"""One-pass split video: original audio, then engine A, then engine B, over the unchanged source video.
Right: one big fact card for the section playing. Below: original spectrogram, playing-audio spectrogram, section bar, playhead.

  python split_video.py OUT.mp4 VIDEO ORIG.wav CUT1 CUT2 "title|number|caption|verdict|out.wav" "title|...|out.wav"
Sections: 0..CUT1 original, CUT1..CUT2 engine A, CUT2..end engine B. Needs ffmpeg.
"""
import os, subprocess, sys, tempfile
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFont

out, video, orig, c1, c2, *eng = sys.argv[1:]
c1, c2 = float(c1), float(c2); eng = [e.split("|") for e in eng]
W, H, FPS = 1920, 1080, 24
VX, VY, VW, VH = 24, 24, 1216, 684
SX, SW, SH, SY0, SY1, BY = 24, 1872, 140, 732, 884, 1032
COL = ["#bbbbbb", "#e0803a", "#2bb673"]
def font(n):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"):
        try: return ImageFont.truetype(p, n)
        except OSError: pass
    return ImageFont.load_default()

def rd(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True); return x[:, 0], sr
x0, sr = rd(orig); n = len(x0); dur = n / sr
tracks = [x0] + [np.pad(rd(e[-1])[0], (0, n))[:n] for e in eng]
# playing audio: section k from track k, 30 ms crossfades at the cuts
g = np.zeros((3, n), np.float32); cuts = [0, int(c1 * sr), int(c2 * sr), n]; f = int(0.03 * sr)
for k in range(3):
    g[k, cuts[k]:cuts[k + 1]] = 1
    if k: g[k, cuts[k] - f:cuts[k] + f] = np.linspace(0, 1, 2 * f); g[k - 1, cuts[k] - f:cuts[k] + f] = np.linspace(1, 0, 2 * f)
play = (g * np.stack(tracks)).sum(0)

def spec(x, ref=None):
    N, HOP = 2048, 256
    fr = np.lib.stride_tricks.sliding_window_view(np.pad(x, (0, N)), N)[::HOP][: len(x) // HOP] * np.hanning(N)
    S = 20 * np.log10(np.abs(np.fft.rfft(fr, axis=1)).T + 1e-6); ref = S.max() if ref is None else ref
    q = Image.fromarray((np.clip((S - (ref - 80)) / 80, 0, 1)[::-1] * 255).astype(np.uint8)).resize((SW, SH))
    return ref, Image.merge("RGB", (q.point(lambda v: min(255, v * 2)), q.point(lambda v: max(0, v * 2 - 200) if v > 100 else 0),
                                    q.point(lambda v: 255 - abs(v * 2 - 255) // 2 if v < 128 else 0)))
ref, s_orig = spec(x0); _, s_play = spec(play, ref)

tmp = tempfile.mkdtemp(); bgs = []
cards = [("Original", "", "Noisy audio, no cleaning", "")] + [tuple(e[:4]) for e in eng]
for k, (title, num, cap, verdict) in enumerate(cards):
    img = Image.new("RGB", (W, H), "#101010"); d = ImageDraw.Draw(img); cx = VX + VW + 40
    d.text((cx, 40), "Same NVIDIA RE-USE weights", font=font(28), fill="#aaaaaa")
    d.text((cx, 76), "8 layers  ·  MacBook Air M3 CPU", font=font(28), fill="#aaaaaa")
    if os.environ.get("NOTE"): d.text((cx, 112), os.environ["NOTE"], font=font(28), fill="#aaaaaa")
    d.text((cx, 170), title, font=font(64), fill=COL[k])
    if num: d.text((cx, 270), num, font=font(110), fill="white"); d.text((cx, 400), cap, font=font(30), fill="#cccccc")
    else: d.text((cx, 270), cap, font=font(36), fill="#cccccc")
    if verdict: d.rounded_rectangle([cx, 470, W - 24, 550], 14, fill=COL[k]); d.text((cx + 22, 488), verdict, font=font(38), fill="black")
    for y, im, lab in ((SY0, s_orig, "Original audio"), (SY1, s_play, "What you hear")):
        img.paste(im, (SX, y)); d.rectangle([SX, y, SX + SW - 1, y + SH - 1], outline="#333333")
        d.rounded_rectangle([SX + 8, y + 6, SX + 20 + 12 * len(lab), y + 36], 8, fill="black"); d.text((SX + 16, y + 9), lab, font=font(22), fill="white")
    for j, (a, b) in enumerate(((0, c1), (c1, c2), (c2, dur))):  # section bar under the spectrograms
        xa, xb = SX + int(SW * a / dur), SX + int(SW * b / dur)
        d.rectangle([xa + 2, BY, xb - 2, BY + 36], fill=COL[j] if j == k else "#222222")
        d.text((xa + 14, BY + 5), cards[j][0], font=font(24), fill="black" if j == k else COL[j])
        if j: d.line([(xa, SY0), (xa, BY + 36)], fill="#666666", width=2)
    p = os.path.join(tmp, f"bg{k}.png"); img.save(p); bgs.append(p)
aw = os.path.join(tmp, "a.wav"); sf.write(aw, play, sr)
fc = (f"[3:v]scale={VW}:{VH},setsar=1,fps={FPS}[v];"
      f"[0:v][1:v]overlay=0:0:enable='between(t,{c1},{c2})'[b1];[b1][2:v]overlay=0:0:enable='gte(t,{c2})'[b2];"
      f"[b2][v]overlay={VX}:{VY}[a];[a][4:v]overlay=x='{SX}+{SW}*t/{dur:.3f}':y={SY0}:eval=frame[o]")
cmd = ["ffmpeg", "-y", "-loglevel", "error"]
for p in bgs: cmd += ["-loop", "1", "-framerate", str(FPS), "-i", p]
cmd += ["-i", video, "-f", "lavfi", "-i", f"color=c=white:s=3x{BY + 36 - SY0}:r={FPS}", "-i", aw,
        "-filter_complex", fc, "-map", "[o]", "-map", "5:a", "-t", f"{dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-threads", "2", "-c:a", "aac", "-b:a", "192k", out]
subprocess.run(cmd, check=True); print(out)
