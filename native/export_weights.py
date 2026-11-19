"""Export RE-USE real-time weights to a flat file for reuse_engine.c.

Format: b"RUSE" u32 count, then per tensor: u32 name_len, name, u32 ndim, u32 dims[ndim], f32 data.
Weights are pre-arranged for row-major sgemm (activations are [rows=freq][cols=channels]):
  conv (co, ci, kt, kf) -> [kt][kf][ci][co];  linear (out, in) -> [in][out];  A_log -> A = -exp(A_log).
"""
import struct, sys
from pathlib import Path
sys.path.insert(0, str(Path.home() / "Coding/reuse-mic"))
import torch
from reuse_core import load_model

out = Path(sys.argv[1] if len(sys.argv) > 1 else "reuse_rt.bin")
model, _ = load_model()
T = {}
for k, v in model.state_dict().items():
    if k.startswith(("mask_decoder.", "phase_decoder.")):  # unused non-list heads
        continue
    v = v.detach().float()
    if k.endswith("A_log"):
        k, v = k[:-5] + "A", -torch.exp(v)
    elif v.ndim == 4:  # conv (co, ci, kh, kw) -> [kh][kw][ci][co]
        v = v.permute(2, 3, 1, 0)
    elif v.ndim == 2 and "proj" in k:  # linear (out, in) -> [in][out]
        v = v.t()
    elif v.ndim == 3:  # depthwise conv1d (d, 1, k) -> [d][k]
        v = v[:, 0, :]
    T[k] = v.contiguous()
with open(out, "wb") as f:
    f.write(b"RUSE" + struct.pack("<I", len(T)))
    for k, v in T.items():
        n = k.encode()
        f.write(struct.pack("<I", len(n)) + n + struct.pack("<I", v.ndim) + struct.pack(f"<{v.ndim}I", *v.shape))
        f.write(v.numpy().astype("<f4").tobytes())
print(out, len(T), "tensors", out.stat().st_size // 1024, "KiB")
