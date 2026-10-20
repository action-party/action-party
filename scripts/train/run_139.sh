#!/usr/bin/env bash
# Train the headline exp-139 coord-token diffusion model.
#
# Prerequisites:
#   1. Melting Pot LMDB dataset at datasets/melting_pot_all_games_512_2k
#      (see docs/DATASET.md for creation / download instructions).
#   2. The exp-138 base-pretrain checkpoint at
#      checkpoints/exp-138-base-wan-pretrain-multi-game-512/checkpoint_model_022500/model.pt
#   3. WAN base weights downloaded to wan_models/Wan2.1-T2V-1.3B
#      (huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B).
#
# Usage:
#   bash scripts/train/run_139.sh [N_GPUS]
set -euo pipefail

N_GPUS="${1:-1}"
CONFIG="configs/exp-139-multi-game-coord-512.yaml"

torchrun --nproc_per_node="${N_GPUS}" --max_restarts=0 \
  train.py \
  --config_path "${CONFIG}"
