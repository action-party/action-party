# 20-action RGB NPZs for per-game tile models

Use `scripts/dataset/create_all_games_dataset.py` with `--num_actions 20` (512 canvas). Each NPZ has **81** `pixel_frames` rows (20 logged actions × 4 substeps + 1).

**Layout for multi-game training:** `NPZ_PARENT/<game>/<game>_s*.npz` (e.g. `datasets/rgb_npz_20act_by_game/clean_up/clean_up_s0.npz`).

**Pipeline (Python entrypoints):**

1. **Gen** — `run_all_mini_games_5act_tile_dual_gpu.py` with `--skip_train --num_actions 20`
2. **Presence** — `train_cell_player_presence_mlp.py` per game (`--holdout_last_seed`, `--first_timestep_only` recommended)
3. **Tile CNNs** — `train_tile_dual_presence_per_game_subdirs.py` (action + cell heads, dual GT presence gate on rows)
4. **Val** — `eval_all_games_val_action_dual_presence.py` or `eval_per_game_action_tile_model.py`

Example shell driver: `example_tile_eval_usage.sh`.

**clean_up reference:** same hyperparams as `rgb_npz_clean_up_20act_n100` — `rgb_stack=current_next`, train-until 95% train acc, Adam 3e-3, hidden 128, batch 256, `precompute_crops`.
