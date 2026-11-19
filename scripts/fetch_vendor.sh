#!/bin/sh
# Downloads NVIDIA's model repos into vendor/ (not committed: NVIDIA licence terms, large weights).
#   vendor/Realtime_REUSE      <- nvidia/Real-time_RE-USE        (NVIDIA One-Way Noncommercial, NSCLv1)
#   vendor/NemotronLabs_REUSE  <- nvidia/NVIDIA-NemotronLabs-RE-USE (OpenMDW-1.1)
set -e
cd "$(dirname "$0")/.."
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/Real-time_RE-USE", local_dir="vendor/Realtime_REUSE")
snapshot_download("nvidia/NVIDIA-NemotronLabs-RE-USE", local_dir="vendor/NemotronLabs_REUSE")
PY
echo "vendor ready. Native weights: cd native && ../.venv/bin/python export_weights.py reuse_rt.bin"
