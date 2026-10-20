#!/bin/bash
# Run full ablation evaluation (loads model ONCE for all samples)
#
# Usage:
#   ./run_ablation_eval.sh <config> <checkpoint> <output_name> [num_samples] [gpu_id]
#
# Examples:
#   ./run_ablation_eval.sh configs/exp-134-coord-token-original-actions.yaml \
#       checkpoints/exp-134-coord-token-original-actions/checkpoint_model_045000/model.pt \
#       baseline-134
#
#   ./run_ablation_eval.sh configs/exp-141-no-self-attn-isolation.yaml \
#       checkpoints/exp-141-no-self-attn-isolation/checkpoint_model_024000/model.pt \
#       no-self-attn-isolation-141 10 1

set -e

CONFIG="$1"
CHECKPOINT="$2"
OUTPUT_NAME="$3"
NUM_SAMPLES="${4:-5}"
GPU_ID="${5:-0}"

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Usage: $0 <config> <checkpoint> <output_name> [num_samples] [gpu_id]"
    exit 1
fi

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/eval/run_ablation_eval.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --name "${OUTPUT_NAME}" \
    --num_samples "${NUM_SAMPLES}"
