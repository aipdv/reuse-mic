# Native RE-USE engine: working notes (2026-11-19)

Goal: run NVIDIA Real-time RE-USE live on an M3 Air (8 GB, fanless) with minimal CPU, no Python.

## Files (this folder)
- `export_weights.py`: PyTorch checkpoint -> `reuse_rt.bin` (49 MB, pre-arranged for row-major sgemm).
- `reuse_engine.c`: the model per STFT frame (`re_create/re_step`) + full hop pipeline (`re_live_create/re_live_process`).
- `reuse_cli.c`: `reuse` binary: `devices`, `live`, `file`, `bench`. Test hooks: `--input-file`, `--feed-rate`, `--tap`.
- `build.sh`: builds `libreuse_prof.dylib` (section timers), `libreuse_engine.dylib`, `reuse`.
- `parity.py` (frame-level vs torch), `parity_live.py` (waveform vs torch), `speed.py`, `prof.py`.

## Verified
- Frame parity vs torch StreamingSEMamba: mag max |d| <= 2e-6, phase <= 4e-5 rad (8-24 kHz, 2-12 layers).
- Waveform parity vs reuse_core.LiveEnhancer: 110 dB SNR (12k/4), 100 dB (16k/8).
- Live plumbing: file fed at model rate through ring/worker/output == direct engine call, bit-exact.
- Device test BlackHole->BlackHole 4 s: 0 late, 0 underruns.

## Speed (single core, per 20 ms hop)
| config | original torch | torch + new scan | native |
|---|---|---|---|
| 12k/4 | 14.8 ms, 1.85 cores | 14.7 ms, 1.14 cores | 6.5 ms, 0.34 cores |
| 16k/4 | 18.3 ms, 2.4 cores | 17.8 ms, 1.5 cores | 8.5 ms, 0.42 cores |
| 16k/8 | over budget | over budget | 17.2 ms |
| 16k/12 | over | over | 25 ms (needs 2 threads) |

Native profile 12k/4: scan 3.7 ms (fused NEON exp), sgemm 1.3, softplus+gate 0.7, conv+silu 0.4, ln 0.15.

## Optimisation log (native)
1. v1 straight port: 15.0 ms (scalar softplus 5.3, vvexpf 3.8, scan 2.6, sgemm 1.4).
2. vectorized softplus (vForce): 12.2.
3. fused NEON scan (inline exp, 16 states in registers): 7.7.
4. NEON LN+PReLU, conv+SiLU fused, transposed conv weights, no copies: 6.84.
5. NEON softplus via 2*atanh series: 6.47.

## Incidents
- Twice the live binary opened the default mic by mistake (stale binary after a failed build ran 40 s; `live --help` ran < 1 s). Nothing saved. Fixed: build gating, `--help`, unknown flags refused.

## Next
- 2-thread split (time rows, fw/bw, mag/pha decoders) for 16k/12 and 24k.
- Install into ~/Coding/reuse-mic/native + README.
- Lower live latency (prefill 3 hops now).

## Update (later the same day)
- 2-thread mode: time-Mamba rows split, freq fw/bw in parallel, mag/pha decoders in parallel. Output bit-identical to 1 thread. About 1.8x faster: 16k/12 = 12.7 ms, 24k/8 = 13.1 ms.
- Live robustness on a busy 8 GB Mac (another heavy job running): late-hop bursts traced to (a) scheduling and (b) compressed/swapped memory.
  Fixes: worker + helper in the Mach real-time class (period = hop, computation = 0.75 hop), all engine memory mlock'ed.
  90 s soaks: 16k/4/1T and 16k/8/2T clean; 16k/12/2T still glitches under heavy outside load.
- Whole-process CPU over 40 s loopback: native 16k/4 = 0.45 core, native 16k/12/2T = 1.22, Python live_mic.py 16k/4 = 1.54 with 500 late hops.
- Long-stream behaviour: on noisy speech the output settles after about 1 min and stays stable (tested 4 min).
  On the music-heavy demo clip the output level wanders (same in PyTorch: a model property). Hard state resets make it worse (loud bursts), so none are applied.
- Defaults: --cushion 3 (about 100 ms). --cushion 2 (about 80 ms) was clean for 16k/4.
- ReuseMic.app (app/): Swift menu bar wrapper around `reuse live`. Self-contained, ad-hoc signed, NSMicrophoneUsageDescription set. Launch smoke-tested only; the engine was not started from it (no mic use while the user was away).
- IO workgroup: worker and helper join the output device's `kAudioOutputUnitProperty_OSWorkgroup` (Apple's pattern for parallel real-time audio threads; `--workgroup 0` disables).
  A/B at load ~1.5: on and off both clean (0 underruns in 90 s at 16k/12/2T). Not proven to help; kept because it is the documented pattern and costs nothing.
