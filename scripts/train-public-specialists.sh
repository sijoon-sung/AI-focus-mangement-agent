#!/usr/bin/env bash
# Run inside a prepared research environment. Optional first argument: cpu / cuda.
set -euo pipefail
cd "$(dirname "$0")/.."
device="${1:-cuda}"
export HF_HOME="$PWD/runs/hf-cache"
export TORCH_HOME="$PWD/runs/torch-cache"
python -m research.public_data mpiigaze
python -m research.public_data groundcua --per-app 1000
python -m research.public_data mind2web --shards 0 1 2 3 4 5 6 7 8 9 10
python -m research.train_specialists --role camera_gaze --per-person 3000 --epochs 40 --device "$device" --output runs/full-public/camera_gaze.pt
python -m research.train_specialists --role screen_layout --epochs 30 --batch-size 32 --device "$device" --output runs/full-public/screen_layout.pt
python -m research.train_specialists --role goal_history --epochs 30 --device "$device" --output runs/full-public/goal_history.pt
