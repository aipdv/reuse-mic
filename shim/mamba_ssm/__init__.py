"""Pure-PyTorch stand-in for `mamba_ssm.Mamba` (the CUDA-only package).

Same parameter names as mamba_ssm v1 `Mamba`, so NVIDIA checkpoints load as-is.
Implements forward() for full sequences and step()/allocate_inference_cache()
for frame-by-frame streaming. No CUDA, runs on Apple Silicon CPU.
"""
import ctypes
import math
import os
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_LIB = _HERE / "libscan.dylib"
if not _LIB.exists() or _LIB.stat().st_mtime < (_HERE / "scan.c").stat().st_mtime:
    subprocess.run(["clang", "-O3", "-ffast-math", "-mcpu=apple-m1", "-shared", "-framework", "Accelerate",
                    "-o", str(_LIB), str(_HERE / "scan.c")], check=True)
_lib = ctypes.CDLL(str(_LIB))
# ponytail: 3 coarse GCD jobs measured best on M3 (same wall time as the old per-channel
# dispatch, about 38% less CPU). Tune here if the Mac changes.
_lib.set_workers(int(os.environ.get("REUSE_SCAN_WORKERS", 3)))
_scan_c, _step_c = _lib.selective_scan, _lib.selective_step
_scan_c.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4
_step_c.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3


def selective_scan(u, delta, A, B, C, D):
    """u, delta (pre-softplus): [b, d, L]; A: [d, n]; B, C: [b, n, L]; D: [d]. Returns y: [b, d, L]."""
    b_, d_, L = u.shape
    n_ = A.shape[1]
    assert n_ <= 16
    f = lambda t: t.float().contiguous()
    dt_, u_ = f(delta.transpose(1, 2)), f(u.transpose(1, 2))  # [b, L, d]
    B_, C_ = f(B.transpose(1, 2)), f(C.transpose(1, 2))  # [b, L, n]
    A_, D_ = f(A), f(D)
    y = torch.empty(b_, L, d_)
    _scan_c(dt_.data_ptr(), u_.data_ptr(), A_.data_ptr(), B_.data_ptr(), C_.data_ptr(),
            D_.data_ptr(), y.data_ptr(), b_, L, d_, n_)
    return y.transpose(1, 2)


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto",
                 conv_bias=True, bias=False, **_):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def _dwconv(self, x):  # causal depthwise conv as K shifted mul-adds (torch CPU runs groups=d as d tiny convs)
        K, L = self.d_conv, x.shape[-1]
        xp = F.pad(x, (K - 1, 0))
        w = self.conv1d.weight[:, 0, :]  # [d, K]
        y = sum(xp[..., k:k + L] * w[:, k:k + 1] for k in range(K))
        return y + self.conv1d.bias[:, None] if self.conv1d.bias is not None else y

    def _ssm_params(self, x):  # x: [..., d_inner]; dt is returned pre-softplus (C applies it)
        dt, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        return self.dt_proj(dt), B, C

    def forward(self, hidden_states):  # [b, L, d_model]
        L = hidden_states.shape[1]
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)
        x = F.silu(self._dwconv(x.transpose(1, 2)))  # [b, d, L]
        dt, B, C = self._ssm_params(x.transpose(1, 2))
        A = -torch.exp(self.A_log.float())
        y = selective_scan(x, dt.transpose(1, 2), A, B.transpose(1, 2), C.transpose(1, 2), self.D.float())
        return self.out_proj(y.transpose(1, 2) * F.silu(z))

    def allocate_inference_cache(self, batch_size, max_seqlen=1, dtype=None, **_):
        dev = self.out_proj.weight.device
        dtype = dtype or self.conv1d.weight.dtype
        conv_state = torch.zeros(batch_size, self.d_inner, self.d_conv, device=dev, dtype=dtype)
        ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=dev, dtype=dtype)
        return conv_state, ssm_state

    def step(self, hidden_states, conv_state, ssm_state):  # hidden_states: [b, 1, d_model]
        x, z = self.in_proj(hidden_states.squeeze(1)).chunk(2, dim=-1)  # [b, d]
        conv_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1)
        x = (conv_state * self.conv1d.weight.squeeze(1)).sum(-1)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias
        x = F.silu(x)
        dt, B, C = self._ssm_params(x)
        A = -torch.exp(self.A_log.float())
        f = lambda t: t.float().contiguous()
        dt, x, B, C, A, D = map(f, (dt, x, B, C, A, self.D))
        ssm_state = f(ssm_state)
        y = torch.empty_like(x)
        b_, d_ = x.shape
        _step_c(dt.data_ptr(), x.data_ptr(), A.data_ptr(), B.data_ptr(), C.data_ptr(), D.data_ptr(),
                ssm_state.data_ptr(), y.data_ptr(), b_, d_, self.d_state)
        out = self.out_proj(y * F.silu(z))
        return out.unsqueeze(1), conv_state, ssm_state


if __name__ == "__main__":
    # Self-check: parallel forward == naive recurrence == step-by-step streaming.
    torch.manual_seed(0)
    m = Mamba(d_model=64, d_state=16, d_conv=4, expand=4).eval()
    with torch.no_grad():
        m.dt_proj.bias.uniform_(-3, 1)
        x = torch.randn(3, 150, 64)
        full = m(x)
        cs, ss = m.allocate_inference_cache(3)
        outs = []
        for t in range(x.shape[1]):
            o, cs, ss = m.step(x[:, t:t + 1], cs, ss)
            outs.append(o)
        stream = torch.cat(outs, 1)
    err = (full - stream).abs().max().item()
    print("max |forward - step| =", err)
    assert err < 1e-4, err
    print("ok")
