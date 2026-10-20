# scripts/eval

## World-model metrics (paper + ablations)

Generate `eval_data/` rollout archives (`.npz` per sample) from a checkpoint, then score them:

| Script | Role |
|--------|------|
| `run_ablation_eval.py` | Load WM once; AR rollout; write `eval_data/`, GIFs; run classifier metrics |
| `eval_classifier_metrics.py` | **Action accuracy**, **detection rate (DR)**, **player preservation** (tile classifier on generated frames) |
| `train_player_classifier.py` | Train `player_tile_classifier.pkl` for the above |
| `finish_ablation_eval_classifier.sh` | Run classifier metrics after sharded `run_ablation_eval` |
| `eval_pixel_quality.py` | Pixel-quality metrics on `eval_data/` |
| `eval_fvd.py` | FVD on `eval_data/` |
| `run_ablation_eval.sh`, `run_ablation_inference.sh` | Batch wrappers |

Example (coins ablation runs): configs `exp-150`–`152`. Drive them with `run_ablation_eval.py` / `run_ablation_eval.sh` as shown above.

## Tile CNN metrics (46 games, RGB episodes)

Per-game tile models and holdout **action / cell / orientation** accuracy on saved RGB episodes (`TILE_RGB_20ACT.md`, `example_tile_eval_usage.sh`):

`train_cell_player_presence_mlp.py`, `train_per_game_action_tile_model.py`, `train_tile_dual_presence_per_game_subdirs.py`, `run_all_mini_games_5act_tile_dual_gpu.py`, `eval_per_game_action_tile_model.py`, `eval_all_games_val_action_dual_presence.py`, `per_game_tile_val_*.py`, plus `mp_action_factorization.py`, `per_game_tile_crops.py`, `per_game_tile_model.py`.
