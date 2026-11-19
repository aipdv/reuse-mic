# RE-USE live mic: optimisation log (M3 Air, 8 GB, on battery, 2026-11-19)

Metric: wall ms per 20 ms hop (median, p95) and CPU ms per hop (energy). Config 12 kHz, 4 layers, 2 torch threads.
Harness: `opt/bench.py` (real speech input, 300 hops, output checked against an eager reference). Raw runs: `ledger.jsonl`.

## Ceiling
- 570 MFLOP/hop (12 kHz, 4 layers); 1,250 MFLOP/hop (16 kHz, 12 layers). Each layer is about 60 MFLOP.
- 1-core GEMM: 830 GFLOPS (AMX). Current code reaches about 45 GFLOPS: overhead bound, not compute bound.

## Experiments
| # | Lever | Result | Verdict |
|---|---|---|---|
| 1 | torch threads 1 / 4 vs 2 | 1 thread: wall +25%; 4: p95 worse | reject |
| 2 | KMP_BLOCKTIME=0, VECLIB_MAXIMUM_THREADS=1 | inside noise | reject |
| 3 | scan kernel serial (no GCD) | CPU -18%, wall +85% | reject |
| 4 | vectorized exp (vvexpf), serial | CPU -41%, wall +30%, bit-exact | superseded by 6 |
| 5 | torch.compile (blocks) | NaN output, no speedup | reject (stateful caches break dynamo) |
| 6 | vvexpf + 3 coarse GCD jobs over channels | wall same (14.7 vs 14.8 ms), CPU -38% (22.7 vs 36.5 ms), bit-exact | **accept, applied** |

Grid with lever 6 (orig -> new CPU ms per hop): 16k/4: 47.9 -> 29.8; 16k/6: 67.1 -> 40.0 (wall 23 ms, over budget); 12k/8: 65.4 -> 38.0 (wall 23 ms, over budget).

## STATUS (plateau)
- Applied to `~/Coding/reuse-mic/shim/mamba_ssm/` (backups: `scan_gcd_orig.c.bak`, `__init__.py.bak`). check.py PASS (cosine 0.99997 offline, 0.998 live vs NVIDIA CUDA).
- Wall time limit now: about 10 ms/hop of PyTorch per-op overhead (cat/pad/copy in streaming convs, slow conv2d path, SiLU, many tiny ops).
- Next levers, ranked:
  1. Ring-buffer causal convs + fused Mamba step in the streaming wrappers (remove cat/pad/fill/copy). Est. -3 to -5 ms/hop. Medium.
  2. Make torch.compile work: scan as a torch custom op, caches as explicit state. Est. 2x on torch side. Medium-high.
  3. Core ML export of the per-hop graph (ANE/GPU), scan stays in C. Biggest energy win. High.
- Not verified: live BlackHole run end to end (needs the mic); runs were on battery with another heavy thread active, so expect about ±10% noise.
