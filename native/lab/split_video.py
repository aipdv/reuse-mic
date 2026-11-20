"""One-pass split video for a general audience: original audio, then engine A, then engine B, over the unchanged video.
Right: section title, one big number, a verdict. Bottom: mel spectrogram of what you hear (sections not playing dimmed) with a playhead.

  python split_video.py OUT.mp4 VIDEO ORIG.wav CUT1 CUT2 "title[~subtitle]|number|caption|verdict|out.wav" "title|...|out.wav"
Sections: 0..CUT1 original, CUT1..CUT2 engine A, CUT2..end engine B. Needs ffmpeg.
"""
import os, subprocess, sys, tempfile
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFont

out, video, orig, c1, c2, *eng = sys.argv[1:]
c1, c2 = float(c1), float(c2); eng = [e.split("|") for e in eng]
W, H, FPS = 1920, 1080, 24
VX, VY, VW, VH = 24, 24, 1312, 738               # video: the main thing
WX, WW, WY, WH, BY = 24, 1872, 800, 170, 984     # waveform strip and section bar
COL = ["#9a9a9a", "#e0803a", "#2bb673"]
def font(n, bold=False):
    for p in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc"):
        try: return ImageFont.truetype(p, n, index=1 if bold else 0)
        except OSError: pass
    return ImageFont.load_default()

def rd(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True); return x[:, 0], sr
x0, sr = rd(orig); n = len(x0); dur = n / sr
tracks = [x0] + [np.pad(rd(e[-1])[0], (0, n))[:n] for e in eng]
g = np.zeros((3, n), np.float32); cuts = [0, int(c1 * sr), int(c2 * sr), n]; f = int(0.03 * sr)
for k in range(3):  # what you hear: section k from track k, 30 ms crossfades
    g[k, cuts[k]:cuts[k + 1]] = 1
    if k: g[k, cuts[k] - f:cuts[k] + f] = np.linspace(0, 1, 2 * f); g[k - 1, cuts[k] - f:cuts[k] + f] = np.linspace(1, 0, 2 * f)
play = (g * np.stack(tracks)).sum(0)

def mel_image(x):  # log-mel spectrogram (128 bands, 0..sr/2), magma-like colours, 80 dB range -> RGB array [WH][WW]
    import torch, torchaudio
    m = torchaudio.transforms.MelSpectrogram(sr, n_fft=2048, hop_length=256, n_mels=128, f_min=20, f_max=sr / 2)(torch.from_numpy(x))
    S = 10 * np.log10(m.numpy() + 1e-10); S = np.clip((S - (S.max() - 80)) / 80, 0, 1)[::-1]
    q = np.array(Image.fromarray((S * 255).astype(np.uint8)).resize((WW, WH), Image.BILINEAR)) / 255.0
    stops = np.array([[0, 0, 4], [59, 15, 112], [140, 41, 129], [222, 73, 104], [254, 159, 109], [252, 253, 191]], float)
    pos = q * (len(stops) - 1); i = np.minimum(pos.astype(int), len(stops) - 2); t = (pos - i)[..., None]
    return (stops[i] * (1 - t) + stops[i + 1] * t).astype(np.uint8)
mel = mel_image(play)

tmp = tempfile.mkdtemp(); bgs = []
cards = [("Original sound", "", "Music and voice, not cleaned", "")] + [tuple(e[:4]) for e in eng]
for k, (title, num, cap, verdict) in enumerate(cards):
    img = Image.new("RGB", (W, H), "#0e0e0e"); d = ImageDraw.Draw(img); cx = VX + VW + 48
    d.text((cx, 40), "Same AI voice cleaner,", font=font(30), fill="#8a8a8a")
    d.text((cx, 78), "two ways to run it", font=font(30), fill="#8a8a8a")
    d.text((cx, 190), "NOW PLAYING", font=font(24, True), fill="#666666")
    title, _, sub = title.partition("~")  # optional subtitle: "Title~subtitle"
    d.text((cx, 222), title, font=font(54, True), fill=COL[k])
    if sub: d.text((cx, 288), sub, font=font(30), fill=COL[k])
    if num:
        d.text((cx, 350), num, font=font(132, True), fill="white")
        d.text((cx, 500), cap, font=font(30), fill="#bbbbbb")
    else:
        d.text((cx, 350), cap, font=font(34), fill="#bbbbbb")
    if verdict:
        d.rounded_rectangle([cx, 580, W - 24, 660], 40, fill=COL[k]); d.text((cx + 34, 598), verdict, font=font(40, True), fill="black")
    for j, (a, b) in enumerate(((0, c1), (c1, c2), (c2, dur))):
        xa, xb = WX + int(WW * a / dur), WX + int(WW * b / dur); on = j == k
        part = mel[:, xa - WX:xb - WX] if on else (mel[:, xa - WX:xb - WX] * 0.28).astype(np.uint8)  # dim the sections not playing
        img.paste(Image.fromarray(part), (xa, WY))
        d.rounded_rectangle([xa + 4, BY, xb - 4, BY + 56], 12, fill=COL[j] if on else "#1c1c1c")
        d.text((xa + 22, BY + 12), cards[j][0].partition("~")[0], font=font(28, on), fill="black" if on else COL[j])
    d.rounded_rectangle([WX + 8, WY + 8, WX + 190, WY + 40], 8, fill="black"); d.text((WX + 18, WY + 12), "What you hear", font=font(22), fill="white")
    p = os.path.join(tmp, f"bg{k}.png"); img.save(p); bgs.append(p)
aw = os.path.join(tmp, "a.wav"); sf.write(aw, play, sr)
fc = (f"[3:v]scale={VW}:{VH},setsar=1,fps={FPS}[v];"
      f"[0:v][1:v]overlay=0:0:enable='between(t,{c1},{c2})'[b1];[b1][2:v]overlay=0:0:enable='gte(t,{c2})'[b2];"
      f"[b2][v]overlay={VX}:{VY}[a];[a][4:v]overlay=x='{WX}+{WW}*t/{dur:.3f}':y={WY - 10}:eval=frame[o]")
cmd = ["ffmpeg", "-y", "-loglevel", "error"]
for p in bgs: cmd += ["-loop", "1", "-framerate", str(FPS), "-i", p]
cmd += ["-i", video, "-f", "lavfi", "-i", f"color=c=white:s=4x{BY + 56 - WY + 10}:r={FPS}", "-i", aw,
        "-filter_complex", fc, "-map", "[o]", "-map", "5:a", "-t", f"{dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-threads", "2", "-c:a", "aac", "-b:a", "192k", out]
subprocess.run(cmd, check=True); print(out)
