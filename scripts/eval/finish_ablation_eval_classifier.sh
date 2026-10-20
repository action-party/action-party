#!/usr/bin/env bash
# Run tile classifier metrics on a run_ablation_eval output directory (after sharded GPU jobs).
#
# Usage:
#   bash scripts/eval/finish_ablation_eval_classifier.sh \
#     outputs_ckpts/ablation_eval/<run-name> \
#     outputs_ckpts/<val-set>/player_tile_classifier.pkl \
#     32 512
set -euo pipefail
REPO="$(cd "$(dirname "$(dirname "$(dirname "$(readlink -f "$0")")")")" && pwd)"
cd "$REPO"

OUT_DIR="${1:?output dir (contains eval_data/ and val_npz/)}"
CLASSIFIER="${2:-outputs_ckpts/val_256/player_tile_classifier.pkl}"
TILE="${3:-32}"
IM="${4:-512}"

python scripts/eval/eval_classifier_metrics.py \
  --gen_dir "$OUT_DIR" \
  --val_dir "$OUT_DIR/val_npz" \
  --classifier "$CLASSIFIER" \
  --tile_size "$TILE" \
  --image_size "$IM" \
  --skip allelopathic_harvest__open

echo "Wrote $OUT_DIR/classifier_action_metrics.{txt,json}"
