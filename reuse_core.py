"""NVIDIA Real-time RE-USE on Apple Silicon CPU (no CUDA).

- Loads the vendored checkpoint with a pure-PyTorch Mamba shim.
- `enhance_offline(wav, sr)`: whole-clip inference (for parity checks).
- `LiveEnhancer`: one 20 ms hop in, one 20 ms hop out (streaming STFT + OLA iSTFT).
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "Realtime_REUSE"
sys.path.insert(0, str(ROOT / "shim"))  # our mamba_ssm stand-in
sys.path.insert(0, str(VENDOR))

from models.stfts import mag_phase_istft, mag_phase_stft  # noqa: E402
from models.streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv import SEMamba_decoder_list  # noqa: E402
from utils.util import load_config  # noqa: E402

RECIPE = next((VENDOR / "recipes").glob("*.yaml"))


def _load_streaming_classes():
    # ponytail: exec NVIDIA's own streaming wrappers instead of copying them; their file
    # also has CUDA-only top-level code, so take only the marked section.
    src = (VENDOR / "online_inference.py").read_text()
    a = src.index("################### Streaming inference modification start")
    b = src.index("################### Streaming inference modification end")
    ns = {"torch": torch, "nn": nn, "F": F}
    exec(compile(src[a:b], "online_inference_streaming", "exec"), ns)
    return ns["StreamingSEMamba"]


StreamingSEMamba = _load_streaming_classes()


def make_even(v):
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


def load_model():
    cfg = load_config(RECIPE)
    model = SEMamba_decoder_list.from_pretrained(str(VENDOR), cfg=cfg).eval()
    return model, cfg


def stft_sizes(cfg, sr):
    base = cfg["stft_cfg"]["sampling_rate"]
    n_fft = make_even(cfg["stft_cfg"]["n_fft"] * sr // base)
    hop = make_even(cfg["stft_cfg"]["hop_size"] * sr // base)
    win = make_even(cfg["stft_cfg"]["win_size"] * sr // base)
    return n_fft, hop, win


@torch.no_grad()
def enhance_offline(model, cfg, wav, sr, exit_layer=8, look_ahead=0):
    """wav: [1, N] float tensor. Mirrors NVIDIA's offline path."""
    n_fft, hop, win = stft_sizes(cfg, sr)
    cf = cfg["model_cfg"]["compress_factor"]
    mag, pha, _ = mag_phase_stft(wav, n_fft=n_fft, hop_size=hop, win_size=win, compress_factor=cf)
    mag_o, pha_o, _ = model(mag, pha, exit_layer, look_ahead)
    m = torch.expm1(F.relu(mag_o))
    mag_o[:, :, ((m == 0).sum(1) / m.shape[1] > 0.5)[0]] = 0  # NVIDIA's sweep-artifact guard
    out = mag_phase_istft(mag_o, pha_o, n_fft, hop, win, cf)
    return F.pad(out, (0, max(0, wav.shape[1] - out.shape[1])))[:, : wav.shape[1]]


class _TapConv(nn.Module):
    """Same output as a (dt, 1)-dilated Conv2d when the time axis holds exactly the
    (kt - 1) * dt + 1 cached frames: keep every dt-th frame, run an undilated conv."""

    def __init__(self, conv):
        super().__init__()
        self.dt = conv.dilation[0]
        self.conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                              stride=conv.stride, padding=conv.padding,
                              dilation=(1, conv.dilation[1]), groups=conv.groups,
                              bias=conv.bias is not None)
        self.conv.load_state_dict(conv.state_dict())

    def forward(self, x):
        return self.conv(x[:, :, :: self.dt])


class LiveEnhancer:
    """Feed `hop` samples at a time; get `hop` enhanced samples back.

    Output lags input by one hop (20 ms) plus look_ahead hops.
    """

    def __init__(self, model, cfg, sr, exit_layer=8, look_ahead=0):
        self.n_fft, self.hop, win = stft_sizes(cfg, sr)
        assert self.n_fft == 2 * self.hop and win == self.n_fft
        self.exit_layer, self.look_ahead = exit_layer, look_ahead
        self.w = torch.hann_window(self.n_fft)
        self.env = self.w[: self.hop] ** 2 + self.w[self.hop:] ** 2  # OLA normaliser at 50% overlap
        self.sm = StreamingSEMamba(model).eval()
        for mod in self.sm.modules():  # torch CPU runs time-dilated convs on a slow path
            if mod.__class__.__name__ == "StreamingCausalConv2d" and mod.conv.dilation[0] > 1:
                mod.conv = _TapConv(mod.conv)
        self.sm.reset((self.n_fft // 2 + 1) // 2 + 1)
        self.in_buf = torch.zeros(self.hop)
        self.tail = torch.zeros(self.hop)
        self.pending = []  # frames collected before the first look-ahead call

    @torch.no_grad()
    def process(self, x):
        """x: 1-D float tensor of length hop. Returns 1-D tensor of length hop."""
        frame = torch.cat([self.in_buf, x]) * self.w
        self.in_buf = x
        spec = torch.fft.rfft(frame)
        mag = torch.log1p(spec.abs())[None, :, None]
        pha = torch.angle(spec)[None, :, None]
        if len(self.pending) < self.look_ahead:  # warm-up for look-ahead
            self.pending.append((mag, pha))
            return torch.zeros(self.hop)
        if self.pending:
            mag = torch.cat([p[0] for p in self.pending] + [mag], -1)
            pha = torch.cat([p[1] for p in self.pending] + [pha], -1)
            self.pending = []
        mag_t, pha_t = self.sm(mag, pha, self.exit_layer, self.look_ahead)
        m = torch.expm1(F.relu(mag_t))
        if (m == 0).float().mean() > 0.5:
            m = m * 0
        y = torch.fft.irfft(torch.polar(m[0], pha_t[0]), n=self.n_fft) * self.w
        out = (self.tail + y[: self.hop]) / self.env
        self.tail = y[self.hop:]
        return out
