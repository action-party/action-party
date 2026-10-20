#!/bin/bash
# Legacy wrapper — per-sample inference lives in run_ablation_eval.py (loads the model once).
#
# Example:
#   bash scripts/eval/run_ablation_eval.sh \
#     configs/exp-141-no-self-attn-isolation.yaml \
#     checkpoints/exp-141-no-self-attn-isolation/checkpoint_model_028500/model.pt \
#     ablation-141 5 0
echo "Use scripts/eval/run_ablation_eval.sh (or run_ablation_eval.py) instead of this script." >&2
exit 1
