"""Live "Enhanced" mic: real mic -> NVIDIA Real-time RE-USE -> BlackHole 2ch.

In Brave / MacParakeet / OBS pick "BlackHole 2ch" as the microphone.
Press Enter to toggle Enhanced <-> Raw (A/B). Ctrl+C to stop.

  .venv/bin/python live_mic.py                       # defaults: 12 kHz, 4 layers
  .venv/bin/python live_mic.py --rate 8000 --layers 6
  .venv/bin/python live_mic.py --seconds 5           # self-test, then exit
"""
import argparse
import queue
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import torch

from reuse_core import LiveEnhancer, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="External Microphone", help="mic name (substring)")
ap.add_argument("--output", default="BlackHole 2ch", help="virtual mic name (substring)")
ap.add_argument("--rate", type=int, default=12000, help="model rate; PortAudio resamples the devices")
ap.add_argument("--layers", type=int, default=4, help="RE-USE exit layer, 3..12 (more = better, slower)")
ap.add_argument("--lookahead", type=int, default=0, help="0..2 frames of 20 ms")
ap.add_argument("--seconds", type=float, default=0, help="run N seconds then print stats (0 = forever)")
args = ap.parse_args()

torch.set_num_threads(2)
model, cfg = load_model()
enh = LiveEnhancer(model, cfg, args.rate, exit_layer=args.layers, look_ahead=args.lookahead)
hop = enh.hop

inq: "queue.Queue[np.ndarray]" = queue.Queue()
out_buf = np.zeros(0, dtype=np.float32)
out_lock = threading.Lock()
PREFILL = 3 * hop  # 60 ms jitter cushion
state = {"enhanced": True, "underruns": 0, "late": 0, "hops": 0, "stop": False}


def on_input(indata, frames, t, status):
    inq.put(indata[:, 0].copy())


def on_output(outdata, frames, t, status):
    global out_buf
    with out_lock:
        if len(out_buf) >= frames:
            outdata[:, 0] = out_buf[:frames]
            out_buf = out_buf[frames:]
        else:
            outdata.fill(0)
            state["underruns"] += 1
    if outdata.shape[1] > 1:
        outdata[:, 1] = outdata[:, 0]


def worker():
    global out_buf
    pending = np.zeros(0, dtype=np.float32)
    with out_lock:
        out_buf = np.zeros(PREFILL, dtype=np.float32)
    while not state["stop"]:
        try:
            pending = np.concatenate([pending, inq.get(timeout=0.2)])
        except queue.Empty:
            continue
        while len(pending) >= hop:
            x, pending = pending[:hop], pending[hop:]
            t0 = time.perf_counter()
            y = enh.process(torch.from_numpy(x)).numpy()  # always run: keeps model state warm for A/B
            if (time.perf_counter() - t0) * 1000 > hop / args.rate * 1000:
                state["late"] += 1
            state["hops"] += 1
            with out_lock:
                out_buf = np.concatenate([out_buf, y if state["enhanced"] else x])


def dev(name, kind):
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d[f"max_{kind}_channels"] > 0:
            return i
    sys.exit(f"no {kind} device matching {name!r}; see: python -m sounddevice")


ins = sd.InputStream(device=dev(args.input, "input"), channels=1, samplerate=args.rate,
                     blocksize=hop, dtype="float32", callback=on_input)
outs = sd.OutputStream(device=dev(args.output, "output"), channels=2, samplerate=args.rate,
                       blocksize=hop, dtype="float32", callback=on_output)
threading.Thread(target=worker, daemon=True).start()
print(f"RE-USE live: {args.input} -> {args.output} | {args.rate} Hz, {args.layers} layers, "
      f"look-ahead {args.lookahead} | latency ~{(2 * hop + PREFILL) / args.rate * 1000 + 20 * args.lookahead:.0f} ms")
with ins, outs:
    if args.seconds:
        time.sleep(args.seconds)
    else:
        print("Enter = toggle Enhanced/Raw, Ctrl+C = stop")
        try:
            for _ in sys.stdin:
                state["enhanced"] = not state["enhanced"]
                print("ENHANCED" if state["enhanced"] else "RAW")
        except KeyboardInterrupt:
            pass
state["stop"] = True
print(f"hops {state['hops']}, late hops {state['late']}, output underruns {state['underruns']}")
