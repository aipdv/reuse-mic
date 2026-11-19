# reuse-mic

An "Enhanced" virtual mic for any app. It cleans your mic live with NVIDIA Real-time RE-USE on the Mac CPU (no CUDA) and sends the result to BlackHole 2ch.

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
cd ~/Coding/reuse-mic
.venv/bin/python live_mic.py            # Enter = toggle Enhanced/Raw, Ctrl+C = stop
```

Then pick **BlackHole 2ch** as the microphone in Brave, MacParakeet, or OBS. Set that app's Mic Mode to Standard.

Flags: `--rate 8000|12000|16000`, `--layers 3..12`, `--lookahead 0..2`, `--input "MacBook Air Microphone"`.
On an M3 Air, 12 kHz with 4 layers runs in real time: about 15 ms per 20 ms hop, about 1.1 CPU cores. 16 kHz with 4 layers fits (about 18 ms, 1.5 cores) with little headroom. 6+ layers at 12 to 16 kHz miss the 20 ms budget.

## Native (no Python, recommended)

`native/reuse` is a 73 KB binary: the same model in C (Accelerate + NEON), about 2 to 3 times less CPU than the Python path, same output (110 dB SNR vs PyTorch).

```bash
cd ~/Coding/reuse-mic/native
./reuse devices                                            # list mics and outputs
./reuse live --input "MacBook Air Microphone"              # 16 kHz, 4 layers: about 0.4 of one core
./reuse live --input "MacBook Air Microphone" --layers 12 --threads 2   # full model, 2 cores
./reuse file in.mp4 out.wav --layers 12 --threads 2        # clean a file (any format CoreAudio reads)
./reuse bench --rate 16000 --layers 8                      # time per 20 ms hop
```

Pick **BlackHole 2ch** as the mic in the meeting app, as above. Enter toggles Enhanced/Raw.

Menu bar app: `native/app/ReuseMic.app` (Start/Stop, Enhanced toggle, Quality, Microphone). First start asks for mic access. Rebuild with `native/app/build_app.sh`.

| Setting | Time per 20 ms hop (M3 Air) | Whole-process CPU | 90 s soak on a busy Mac |
|---|---|---|---|
| 16 kHz, 4 layers, 1 thread (**daily**) | 8.5 ms | 0.45 core | 0 glitches (also with `--cushion 2`) |
| 16 kHz, 8 layers, 2 threads (**high**) | 8.9 ms | about 0.9 core | 0 glitches |
| 16 kHz, 12 layers, 2 threads | 12.7 ms | 1.2 cores | some glitches when other apps load the CPU |
| Python `live_mic.py`, 16 kHz, 4 layers | 15+ ms | 1.5 cores | 500 late hops |

The worker runs in the macOS real-time class and all model memory is pinned (mlock), so a busy or swapping Mac does not stall it. Latency is about 100 ms (`--cushion 3`, default) or about 80 ms (`--cushion 2`).

Rebuild: `./build.sh`. Re-export weights: `../.venv/bin/python export_weights.py reuse_rt.bin`. Checks: `parity.py`, `parity_live.py`. Details: `NOTES.md`.

## Check

`.venv/bin/python check.py` compares this port with NVIDIA's CUDA reference outputs and measures speed per hop.

## Where

- `reuse_core.py`: model loading, offline enhancement, and the live frame-by-frame enhancer.
- `shim/mamba_ssm/`: pure-PyTorch Mamba with a C scan kernel (vectorized Accelerate `vvexpf`, 3 coarse GCD jobs; `REUSE_SCAN_WORKERS` overrides). Replaces the CUDA-only package. The old per-channel kernel is kept as `scan_gcd_orig.c.bak`.
- `vendor/Realtime_REUSE/`: NVIDIA checkpoint and code, unchanged. Not committed; `scripts/fetch_vendor.sh` downloads it.
- `opt/`: PyTorch-path optimisation log (`PLAN.md`), bench harness, rejected scan-kernel variants.
- `tools/make_ab.py`: in-place A/B/C comparison video (one clip, audio switches between models).

License: the model is NVIDIA One-Way Noncommercial (NSCLv1). Use it for personal demos only.
