#!/usr/bin/env bash
# Example commands for the tile + presence pipeline (20-action RGB NPZs, per-game subdirs).
# Copy/adapt paths and env vars. See TILE_RGB_20ACT.md for NPZ layout and conventions.
#
# From repo root:
#   bash scripts/eval/example_tile_eval_usage.sh          # runs STEP 3 only (dual-presence sweep)
#   RUN_ALL=1 bash scripts/eval/example_tile_eval_usage.sh  # gen + presence + sweep (long)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

NPZ_PARENT="${NPZ_PARENT:-datasets/rgb_npz_20act_by_game}"
PRESENCE_DIR="${PRESENCE_DIR:-outputs/presence}"
OUT_ROOT="${OUT_ROOT:-outputs/tile_dual_presence}"
DUAL_TEMPLATE="${DUAL_TEMPLATE:-${PRESENCE_DIR}/{game}_cell_presence_mlp.pt}"

step1_gen_npz() {
  python3 scripts/eval/run_all_mini_games_5act_tile_dual_gpu.py \
    --skip_train --num_actions 20 --samples_per_game 100 --seed 42 \
    --npz_parent "${NPZ_PARENT}" --num_workers 4 --gpu_ids 0 --resume_npz
}

step2_presence_all_games() {
  mkdir -p "${PRESENCE_DIR}"
  for game in "${NPZ_PARENT}"/*/; do
    game="$(basename "${game%/}")"
    sub="${NPZ_PARENT}/${game}"
    n="$(find "${sub}" -maxdepth 1 -name '*.npz' | wc -l)"
    [[ "${n}" -ge 2 ]] || continue
    out="${DUAL_TEMPLATE//\{game\}/${game}}"
    python3 scripts/eval/train_cell_player_presence_mlp.py \
      --val_dir "${sub}" --basename_contains "${game}" --holdout_last_seed \
      --first_timestep_only --cell_px 32 --image_size 512 \
      --pad_mode constant --pad_color 255,0,255 \
      --epochs 40 --batch 256 --lr 1e-3 --hidden 256 --neg_per_pos 2 --workers 0 \
      --output "${out}"
  done
}

step3_dual_presence_tile_sweep_2gpu() {
  mkdir -p "${OUT_ROOT}"
  args=(
    --npz_parent "${NPZ_PARENT}"
    --out_root "${OUT_ROOT}"
    --dual_presence_ckpt_template "${DUAL_TEMPLATE}"
    --num_shards 2
    --train_until_train_acc_pct 95
    --max_epochs 3000 --lr 3e-3 --batch 256 --hidden 128 --workers 0
  )
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval/train_tile_dual_presence_per_game_subdirs.py \
    "${args[@]}" --shard_id 0 &
  CUDA_VISIBLE_DEVICES=1 python3 -u scripts/eval/train_tile_dual_presence_per_game_subdirs.py \
    "${args[@]}" --shard_id 1 &
  wait
}

# Optional: batch val action acc after checkpoints exist
# python3 scripts/eval/eval_all_games_val_action_dual_presence.py \
#   --npz_parent "${NPZ_PARENT}" \
#   --action_ckpt_template "${OUT_ROOT}/{game}/action.pt" \
#   --dual_presence_ckpt_template "${DUAL_TEMPLATE}"

if [[ "${RUN_ALL:-0}" == "1" ]]; then
  step1_gen_npz
  step2_presence_all_games
fi
step3_dual_presence_tile_sweep_2gpu
