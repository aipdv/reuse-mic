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

## 48 kHz full band (2026-11-19, evening)
Goal: the model's full 48 kHz band live (before: 12/16 kHz, everything above 6/8 kHz cut). 48 kHz = 961 bins, 3x the 16 kHz rows.

| config (M3 Air, live soak 60 s via BlackHole, paced) | median | p95 | underruns | CPU |
|---|---|---|---|---|
| before: 48k/4, 2 threads (bench) | 19.4 ms | 26.8 ms | not real time | |
| 48k/4, 4 threads | 8.7 ms | 8.9 ms | 0 | 1.76 cores |
| 48k/4, 3 threads | 10.9 ms | 11.1 ms | 5 | 1.66 cores |
| 48k/4, 2 threads | 13.3 ms | 27 ms | 618 (RT demotion bursts) | |
| 32k/4, 2 threads | 9.2 ms | 9.5 ms | 0 | 0.94 cores |
| 24k/4, 2 threads | 7.3 ms | 7.7 ms | 0 | 0.74 cores |
| 16k/4, 1 thread (bench) | 7.55 ms (was 8.6) | 8.2 ms | | 0.38 cores (was 0.45) |

Parity: 106 dB vs the previous engine on the demo clip; 76.8 dB vs PyTorch LiveEnhancer at 48k (`parity_live.py 48000 4`, workspace venv has torchaudio).
Quality: `lab/spec.py` spectrograms: 48k keeps clean sibilants/consonants up to ~20 kHz; 12k output stops at 6 kHz.

Kernel log (single core, 48k/4: 24.8 -> 23.5 ms per hop; scan+softplus+gate 17.5 -> ~14 ms):
1. Calibrated the core: 8.6 G vector FMA/s (`lab/peak.c`). The scan needs ~60 FP ops per channel-step, so it already ran near the FP ceiling: only op cuts help.
2. exp -> exp2 with A pre-scaled by log2(e), shift trick (no frintn/fcvtzs), exponent added as an integer, no clamp:
   a per-channel bound `vlim[d] = 125 / max|A2[d]|` guards the fast path; the clamped path runs when any dt in a 4-channel group exceeds it (0% on real data).
   `-ffast-math` folds `(t + S) - S` to `t` and `t - n` to 0: the asm register barriers in `v_exp2_fast` are required.
   Degree 4 was 8% faster but 72 dB parity; degree 5 keeps 106 dB. Kept degree 5.
3. 4 channels per iteration, readout of 4 channels with one pairwise-add tree, D*u and the SiLU gate fused into the store.
4. Softplus fused inside the scan loop made it 2x SLOWER: its latency heads every exp chain and the out-of-order window
   cannot cover enough channel groups. Fix: a per-row pre-pass (softplus + silu(z) into 1 KB L1 buffers), then the scan.
5. fdiv -> reciprocal estimate + 2 Newton steps in SiLU/softplus.
6. Real FFT (vDSP zrop) instead of DFT matrices when n_fft allows (8..48 kHz): 48k matrices were 29 MB pinned and 29 MB read per hop.

Threads: persistent pool (`pfor`), up to 4 threads, ~36 regions per hop: row chunks (encoder/decoder convs, time Mamba,
Mamba projections) and channel chunks for the sequential frequency scans (fw/bw x 4 x 64 channels). CAS ticket tagged by
region so late workers never run stale tasks. Workers spin 0.3 ms between regions, then sleep. Output is the same for any
thread count (132 dB; sgemm row chunking only). Single thread uses one chunk (bigger sgemm calls).
- sgemm (AMX) scales across threads: 527 -> 900 -> 1059 GFLOPS for 1/2/4 threads (`lab/amx_share.c`).
- Summed task time grows ~1.6x at 4 threads (clock drop, shared AMX/L2): 4 threads = 1.7-1.8 cores at 48k.
- Real-time class trap: a hop over the declared 15 ms computation demotes the thread (bursts of late hops). The bench now
  paces hops every 20 ms like live audio and runs in the real-time class (`--realtime 0` to disable); back-to-back RT hops
  get demoted and measure 5x slower. `prof_regions.py` runs without RT for per-region work numbers.

Next lever (not done): the scan is ~60% of the work and sits at the CPU FP ceiling. Offloading the scans to the GPU (Metal,
unified memory, 4096 lanes per scan) could cut CPU at 48k from ~1.8 to well under 1 core, at the cost of GPU scheduling jitter.
