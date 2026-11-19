"""Per-hop latency + CPU cost of RE-USE LiveEnhancer.

  python opt/bench.py --rate 16000 --layers 4 --threads 1 [--hops 300] [--profile]
Prints one JSON line (also appended to opt/ledger.jsonl).
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.home() / "Coding/reuse-mic"))
import numpy as np, torch
from reuse_core import LiveEnhancer, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--rate", type=int, default=12000)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--hops", type=int, default=300)
ap.add_argument("--tag", default="baseline")
ap.add_argument("--profile", action="store_true")
ap.add_argument("--compile", default="", help="blocks|all")
ap.add_argument("--scan-workers", type=int, default=0, help="0 = original GCD kernel")
a = ap.parse_args()
torch.set_num_threads(a.threads)
if a.scan_workers:
    import ctypes, mamba_ssm
    lib = ctypes.CDLL(str(Path(__file__).parent / "scan" / __import__("os").environ.get("SCANLIB", "libscan3.dylib")))
    lib.set_workers(a.scan_workers)
    mamba_ssm._scan_c, mamba_ssm._step_c = lib.selective_scan, lib.selective_step
    mamba_ssm._scan_c.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4
    mamba_ssm._step_c.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 3
model, cfg = load_model()
enh = LiveEnhancer(model, cfg, a.rate, exit_layer=a.layers)
def maybe_compile(e):
    if a.compile == "all":
        e.sm.forward = torch.compile(e.sm.forward, dynamic=False)
    elif a.compile == "blocks":
        for blk in list(e.sm.mamba) + [e.sm.encoder] + list(e.sm.mag_decoder_list) + list(e.sm.pha_decoder_list):
            blk.forward = torch.compile(blk.forward, dynamic=False)
    return e
maybe_compile(enh)
_t0 = time.perf_counter()
import soundfile as sf, torchaudio.functional as AF
wav, sr0 = sf.read(Path.home() / "Coding/reuse-mic/vendor/Realtime_REUSE/noisy_audio/mic_test.wav", dtype="float32")
wav = AF.resample(torch.from_numpy(wav), sr0, a.rate)
need = (a.hops + 20) * enh.hop
wav = wav.repeat(need // len(wav) + 1)[:need]  # loop real noisy speech
x = wav.view(-1, enh.hop)
for i in range(20):
    enh.process(x[i])  # warm-up (+ compile)
compile_s = round(time.perf_counter() - _t0, 1)
ref = Path(__file__).parent / f"ref_{a.rate}_{a.layers}.pt"
if not a.profile:
    e2 = LiveEnhancer(model, cfg, a.rate, exit_layer=a.layers)
    if a.compile and ref.exists(): maybe_compile(e2)
    y = torch.cat([e2.process(x[i]) for i in range(200)])
    if not ref.exists():
        torch.save(y, ref)
    r0 = torch.load(ref)
    maxdiff = float((y - r0).abs().max()); snr = float(10 * torch.log10(r0.pow(2).sum() / ((y - r0).pow(2).sum() + 1e-20)))
if a.profile:
    from torch.profiler import profile
    with profile() as p:
        for i in range(20, 120):
            enh.process(x[i])
    print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
    sys.exit()
wall, cpu = [], []
for i in range(20, 20 + a.hops):
    w0, c0 = time.perf_counter(), time.process_time()
    enh.process(x[i])
    wall.append(time.perf_counter() - w0); cpu.append(time.process_time() - c0)
w, c = np.array(wall) * 1e3, np.array(cpu) * 1e3
hop_ms = enh.hop / a.rate * 1e3
r = dict(tag=a.tag, rate=a.rate, layers=a.layers, threads=a.threads, hop_ms=hop_ms,
         med_ms=round(float(np.median(w)), 2), p95_ms=round(float(np.percentile(w, 95)), 2),
         max_ms=round(float(w.max()), 2), cpu_ms=round(float(np.median(c)), 2),
         cpu_load=round(float(np.median(c)) / hop_ms, 3), scan_workers=a.scan_workers, compile=a.compile, compile_s=compile_s, maxdiff=maxdiff, snr_db=round(snr, 1), t=time.strftime("%H:%M:%S"))
print(json.dumps(r))
with open(Path(__file__).parent / "ledger.jsonl", "a") as f:
    f.write(json.dumps(r) + "\n")
