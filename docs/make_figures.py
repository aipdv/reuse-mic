"""README figures: uv run --with matplotlib --with soundfile docs/make_figures.py RAW.wav CLEAN.wav
Numbers are copied from the README tables (M3 MacBook Air). Edit them here when the README changes."""
import sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "docs"
ENGINE, TORCH, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 11, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": BG})


def deadline(ax, top):
    ax.axvline(20, color=INK2, ls="--", lw=1.2, zorder=1)
    ax.text(19.4, top, "20 ms real-time deadline", color=INK2, fontsize=9, va="center", ha="right")


def clean(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(left=False)


def speed():
    # README "Speed vs NVIDIA's code": ms to clean each 20 ms hop, 48 kHz, same clip. (layers, torch, engine, engine p95)
    rows = [("4 layers", 43.0, 8.6, 8.8), ("8 layers", 63.6, 14.1, 28.0)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y, h = np.arange(len(rows)), 0.36
    for i, (_, t, e, p95) in enumerate(rows):
        ax.barh(i - h / 2 - 0.02, t, h, color=TORCH)
        ax.barh(i + h / 2 + 0.02, e, h, color=ENGINE, xerr=[[0], [p95 - e]], error_kw={"ecolor": INK2, "capsize": 4, "lw": 1.2})
        ax.text(t + 0.8, i - h / 2 - 0.02, f"{t:.1f} ms", va="center", fontsize=10)
        note = "live-safe" if p95 < 20 else "p95 misses the deadline"
        ax.text(p95 + 1.2, i + h / 2 + 0.02, f"{e:.1f} ms, p95 {p95:g} ms  ({t / e:.1f}x faster, {note})", va="center", fontsize=9,
                bbox={"fc": BG, "ec": "none", "pad": 1.5}, zorder=3)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_ylim(1.55, -0.95)
    ax.set_xlim(0, 80)
    ax.set_xlabel("Time to clean 20 ms of audio (lower is better)")
    deadline(ax, -0.7)
    clean(ax)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TORCH), plt.Rectangle((0, 0), 1, 1, color=ENGINE)],
              labels=["NVIDIA PyTorch code (CPU shim)", "reuse-mic native engine"], frameon=False,
              loc="upper right", fontsize=9)
    ax.set_title("4 to 5x faster than NVIDIA's code on the same Mac CPU", loc="left", fontsize=13, pad=14)
    fig.savefig(f"{OUT}/speed.png")


def settings():
    # README "Native" settings table: ms per hop and whole-process CPU per setting.
    rows = [("Daily: 16 kHz, 4 layers", 7.6, 0.4, ENGINE), ("High: 16 kHz, 8 layers", 8.7, 0.9, ENGINE),
            ("Wide band: 32 kHz, 4 layers", 9.2, 0.95, ENGINE), ("Full band: 48 kHz, 4 layers", 8.7, 1.8, ENGINE),
            ("Python live_mic.py: 16 kHz, 4 layers", 15.0, 1.5, TORCH)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(rows))
    for i, (_, ms, cpu, c) in enumerate(rows):
        a.barh(i, ms, 0.6, color=c)
        b.barh(i, cpu, 0.6, color=c)
        a.text(ms + 0.3, i, f"{ms:g}+ ms" if c == TORCH else f"{ms:g} ms", va="center", fontsize=9)
        b.text(cpu + 0.03, i, f"{cpu:g} cores", va="center", fontsize=9)
    a.set_yticks(y, [r[0] for r in rows])
    a.set_ylim(4.5, -1.1)
    a.set_xlim(0, 27)
    b.set_xlim(0, 2.2)
    a.set_xlabel("Time per 20 ms hop")
    b.set_xlabel("Whole-process CPU (cores)")
    deadline(a, -0.75)
    clean(a), clean(b)
    fig.suptitle("Every native setting fits in 20 ms; pick by CPU budget", x=0.125, ha="left", fontsize=13)
    fig.savefig(f"{OUT}/settings.png")


def spectrogram(raw, out):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"hspace": 0.3})
    for ax, path, title in zip(axes, (raw, out), ("Raw mic input", "After reuse-mic (native engine, 48 kHz, 4 layers)")):
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x[:, 0]
        ax.specgram(x, NFFT=2048, Fs=sr, noverlap=1792, cmap="magma", vmin=-130, vmax=-30, window=np.hanning(2048))
        ax.set_ylim(0, sr / 2)
        ax.set_yticks([0, 4000, 8000, 12000, 16000, 20000], ["0", "4", "8", "12", "16", "20"])
        ax.set_ylabel("kHz")
        ax.set_title(title, loc="left", fontsize=11)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[1].set_xlabel("Seconds")
    fig.suptitle("Same clip, before and after: the noise between words is gone", x=0.125, ha="left", fontsize=13)
    fig.savefig(f"{OUT}/spectrogram.png")


if __name__ == "__main__":
    speed()
    settings()
    if len(sys.argv) == 3:
        spectrogram(sys.argv[1], sys.argv[2])
