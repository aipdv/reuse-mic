# reuse-mic

An "Enhanced" virtual mic for any app. It cleans your mic live with NVIDIA Real-time RE-USE on the Mac CPU (no CUDA) and sends the result to BlackHole 2ch.

## Model

- [NVIDIA Real-time RE-USE](https://huggingface.co/nvidia/Real-time_RE-USE) (streaming), model version `12_random_layer_ahead_sep_conv2_1010k`. Not the offline [RE-USE](https://huggingface.co/nvidia/RE-USE).
- 3.7M parameters at the full 12 layers; a setting runs only part of it (2.0M at 4 layers, 2.9M at 8).
- Weights are used unchanged. This repo rewrites the engine that runs them (C, Accelerate, NEON), not the model.

## Speed vs NVIDIA's code (M3 MacBook Air, CPU, 4 threads each)

| | NVIDIA's PyTorch code | This engine |
|---|---|---|
| Time to clean each 20 ms of audio (8 layers) | 61.9 ms (44.1 kHz), not real time | 14.1 ms (48 kHz), 4.4x faster |
| Output difference vs PyTorch | | 77 dB SNR at 48 kHz, 110 dB at 12 kHz (inaudible) |

NVIDIA's code needs CUDA, so the PyTorch number runs it with the CPU Mamba shim in `shim/`. Details: `native/NOTES.md`.

## Setup (fresh clone)

Weights, NVIDIA code and built binaries are not in git. Recreate them:

```bash
uv venv -p 3.12 .venv && uv pip install -p .venv torch numpy soundfile sounddevice einops pyyaml safetensors huggingface_hub
scripts/fetch_vendor.sh                                    # NVIDIA repos -> vendor/
cd native && ../.venv/bin/python export_weights.py reuse_rt.bin && ./build.sh   # native weights + binary
app/build_app.sh                                           # ReuseMic.app (optional)
```

Install [BlackHole 2ch](https://github.com/ExistentialAudio/BlackHole) for the virtual mic.

## Run

```bash
.venv/bin/python live_mic.py            # Enter = toggle Enhanced/Raw, Ctrl+C = stop
```

Then pick **BlackHole 2ch** as the microphone in Brave, MacParakeet, or OBS. Set that app's Mic Mode to Standard.

Flags: `--rate 8000|12000|16000`, `--layers 3..12`, `--lookahead 0..2`, `--input "MacBook Air Microphone"`.
On an M3 Air, 12 kHz with 4 layers runs in real time: about 15 ms per 20 ms hop, about 1.1 CPU cores. 16 kHz with 4 layers fits (about 18 ms, 1.5 cores) with little headroom. 6+ layers at 12 to 16 kHz miss the 20 ms budget.

## Native (no Python, recommended)

`native/reuse` is a 91 KB binary: the same model in C (Accelerate + NEON), about 4 times less CPU than the Python path, same output (77 to 110 dB SNR vs PyTorch, inaudible).

```bash
cd native
./reuse devices                                            # list mics and outputs
./reuse live --input "MacBook Air Microphone"              # 16 kHz, 4 layers: about 0.4 of one core
./reuse live --input "MacBook Air Microphone" --rate 48000           # full band (up to 24 kHz): 4 threads, about 1.8 cores
./reuse live --input "MacBook Air Microphone" --rate 32000           # wide band: 2 threads, about 0.95 core
./reuse live --input "MacBook Air Microphone" --layers 12 --threads 4   # full model depth, about 1.5 cores
./reuse file in.mp4 out.wav --layers 12 --threads 4        # clean a file (any format CoreAudio reads)
./reuse bench --rate 48000 --layers 4                      # time per 20 ms hop and CPU (paced like live)
```

Pick **BlackHole 2ch** as the mic in the meeting app, as above. Enter toggles Enhanced/Raw.

Menu bar app: `native/app/ReuseMic.app` (Start/Stop, Enhanced toggle, Quality incl. Wide band 32 kHz and Full band 48 kHz, Microphone). First start asks for mic access. Rebuild with `native/app/build_app.sh`.

`--threads` defaults by rate: 1 up to 16 kHz, 2 up to 32 kHz, 4 above. Higher rates keep the top of the voice
(consonants, "air"): 16 kHz output stops at 8 kHz, 48 kHz output reaches about 20 kHz.

| Setting | Time per 20 ms hop (M3 Air) | Whole-process CPU | Soak (60 to 90 s) |
|---|---|---|---|
| 48 kHz, 4 layers, 4 threads (**full band**) | 8.7 ms | 1.8 cores | 0 glitches |
| 32 kHz, 4 layers, 2 threads (**wide band**) | 9.2 ms | 0.95 core | 0 glitches |
| 16 kHz, 4 layers, 1 thread (**daily**) | 7.6 ms | 0.4 core | 0 glitches (also with `--cushion 2`) |
| 16 kHz, 8 layers, 2 threads (**high**) | 8.7 ms | about 0.9 core | 0 glitches |
| 16 kHz, 12 layers, 4 threads | 7.5 ms | 1.5 cores | bench only |
| Python `live_mic.py`, 16 kHz, 4 layers | 15+ ms | 1.5 cores | 500 late hops |

The worker runs in the macOS real-time class and all model memory is pinned (mlock), so a busy or swapping Mac does not stall it. Latency is about 100 ms (`--cushion 3`, default) or about 80 ms (`--cushion 2`).

Rebuild: `./build.sh`. Kernel tools and A/B helpers: `native/lab/` (scan microbenchmark, spectrograms, SNR). Re-export weights: `../.venv/bin/python export_weights.py reuse_rt.bin`. Checks: `parity.py`, `parity_live.py`. Details: `NOTES.md`.

## Check

`.venv/bin/python check.py` compares this port with NVIDIA's CUDA reference outputs and measures speed per hop.

## Where

- `reuse_core.py`: model loading, offline enhancement, and the live frame-by-frame enhancer.
- `shim/mamba_ssm/`: pure-PyTorch Mamba with a C scan kernel (vectorized Accelerate `vvexpf`, 3 coarse GCD jobs; `REUSE_SCAN_WORKERS` overrides). Replaces the CUDA-only package. The old per-channel kernel is kept as `scan_gcd_orig.c.bak`.
- `vendor/Realtime_REUSE/`: NVIDIA checkpoint and code, unchanged. Not committed; `scripts/fetch_vendor.sh` downloads it.
- `opt/`: PyTorch-path optimisation log (`PLAN.md`), bench harness, rejected scan-kernel variants.
- `tools/make_ab.py`: in-place A/B/C comparison video (one clip, audio switches between models).

License: the code in this repo is MIT (see `LICENSE`). The model weights are NVIDIA One-Way Noncommercial (NSCLv1) and are not included; use them for personal demos only.
