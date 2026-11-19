"""Run RE-USE variants on NVIDIA's Hugging Face Spaces (ZeroGPU), not locally.

  .venv/bin/python run_spaces.py demo/standard.wav demo/spaces
"""
import shutil
import sys
from pathlib import Path

from gradio_client import Client, handle_file

inp, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

# (output name, space, extra args). target_sr "" = run at the file's own rate (48 kHz).
RUNS = [
    ("nemotron_48k", "nvidia/NVIDIA-NemotronLabs-RE-USE", dict(lp_sr="", target_sr="")),
    ("nemotron_24k", "nvidia/NVIDIA-NemotronLabs-RE-USE", dict(lp_sr="", target_sr="24000")),
    ("nemotron_16k", "nvidia/NVIDIA-NemotronLabs-RE-USE", dict(lp_sr="", target_sr="16000")),
    ("realtime_48k_L8", "nvidia/Real-time_RE-USE", dict(exit_layer="8", look_ahead_frames="0", lp_sr="", target_sr="")),
    ("realtime_48k_L12_LA2", "nvidia/Real-time_RE-USE", dict(exit_layer="12", look_ahead_frames="2", lp_sr="", target_sr="")),
]

clients = {}
for name, space, kw in RUNS:
    dst = out_dir / f"{name}.wav"
    if dst.exists():
        print(f"skip {name} (exists)")
        continue
    c = clients.setdefault(space, Client(space, verbose=False))
    try:
        _, enhanced, _ = c.predict(audio_path=handle_file(str(inp)), video_path=None,
                                   api_name="/unified_enhance", **kw)
        shutil.copy(enhanced, dst)
        print(f"ok   {name} -> {dst}")
    except Exception as e:  # keep going; one failed variant should not lose the others
        print(f"FAIL {name}: {e}")
