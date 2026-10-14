#!/bin/bash
#SBATCH --job-name=focus-research
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=slurm-%A_%a.out
set -euo pipefail
# Run sbatch from repository root after activating the environment.
# Adapt partition/account/GPU syntax to your lab; this template uses one GPU per run.
: "${MANIFEST:?Export MANIFEST=/absolute/path/daisee.csv before sbatch}"
seeds=(42 43 44)
models=(tiny_mean tiny_tcn tiny_tcn resnet18_tcn)
experiment=$((SLURM_ARRAY_TASK_ID / 3))
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}
model=${models[$experiment]}
extra=()
if [ "$experiment" -ge 2 ]; then extra+=(--weighted-loss); fi
if [ "$experiment" -eq 3 ]; then extra+=(--pretrained); fi
python -m research.train --manifest "$MANIFEST" \
  --model "$model" --seed "$seed" --epochs 30 --workers 4 --device cuda \
  --out "runs/experiment${experiment}_seed${seed}" "${extra[@]}"
# Test is deliberately a separate command after the protocol is frozen.
