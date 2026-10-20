# ActionParty: Multi-Subject Action Binding in Generative Video Games

Alexander Pondaven, Ziyi Wu, Igor Gilitschenski, Philip Torr, Sergey Tulyakov, Fabio Pizzati, Aliaksandr Siarohin

[[Project Page](https://action-party.github.io/)] [[arXiv](https://arxiv.org/abs/2604.02330)]

> Recent advances in video diffusion have enabled the development of "world models" capable of simulating interactive environments. However, these models are largely restricted to single-agent settings, failing to control multiple agents simultaneously in a scene. In this work, we tackle a fundamental issue of **action binding** in existing video diffusion models, which struggle to associate specific actions with their corresponding subjects. We propose **ActionParty**, an action-controllable multi-subject world model for generative video games. It introduces *subject state* tokens — latent variables that persistently capture the state of each subject in the scene. By jointly modeling state tokens and video latents with a spatial biasing mechanism, we disentangle global video frame rendering from individual action-controlled subject updates. We evaluate ActionParty on the Melting Pot benchmark, demonstrating the first video world model capable of controlling **up to seven players** simultaneously across **46 diverse environments**.

## Code

This repository implements ActionParty training and evaluation on [Melting Pot](https://github.com/google-deepmind/meltingpot) (512×512, 46 mini-map games), built on [Self-Forcing](https://github.com/guandeh17/Self-Forcing) and [Wan2.1 T2V-1.3B](https://github.com/Wan-Video/Wan2.1). Implementation details (coord tokens, spatial-RoPE cross-attention, subject-isolated self-attention) are in [`docs/METHOD.md`](docs/METHOD.md). Checkpoints and datasets are not included (`outputs/`, `datasets/`, `wan_models/`).

### Repository layout

```
configs/          # experiment training configs (131–147, 150–154, 157–159)
docs/             # METHOD, DATASET, TRAINING, ABLATIONS
model/            # conditioners for multi-agent experiments
pipeline/         # training / inference pipelines
scripts/          # dataset, inference, train, eval
trainer/          # training loops
utils/            # datasets, wan wrapper, game descriptions
wan/              # Wan DiT + coord-token extensions
train.py
```

### What we added

**Coord-token Wan model**

- New: `wan/modules/coord_token.py`, `wan/modules/spatial_cross_attn.py`
- Updated: `wan/modules/model.py`, `wan/modules/causal_model.py`, `train.py`, `trainer/diffusion.py`, `utils/wan_wrapper.py`, `utils/dataset.py`, training/inference `pipeline/` code

**Data and training**

- `utils/game_descriptions.py`, `utils/position_maps.py`, `utils/attention_logger.py`
- `model/` conditioners (`multi_subject*.py`, etc.)
- `scripts/dataset/create_all_games_dataset.py`, `scripts/dataset/mini_maps.py`
- `configs/exp-*.yaml` (131–147), `docs/METHOD.md`, `docs/DATASET.md`, `docs/TRAINING.md`, `docs/ABLATIONS.md`
- `scripts/inference/inference_all_games_coord.py` (qualitative rollouts and comparison GIFs)

**Eval** (`scripts/eval/`, see `scripts/eval/README.md`):

- **World-model:** `run_ablation_eval.py`, `eval_classifier_metrics.py` (action accuracy, detection rate, player preservation), `train_player_classifier.py`, `eval_pixel_quality.py`, `eval_fvd.py`, `finish_ablation_eval_classifier.sh`
- **Tile-based eval** (crop CNNs on saved RGB episodes): `train_cell_player_presence_mlp.py`, `train_per_game_action_tile_model.py`, `train_tile_dual_presence_per_game_subdirs.py`, `eval_per_game_action_tile_model.py`, `eval_all_games_val_action_dual_presence.py`, `per_game_tile_val_*.py`, `run_all_mini_games_5act_tile_dual_gpu.py`, `mp_action_factorization.py`, `per_game_tile_crops.py`, `per_game_tile_model.py`

Additional ablation configs: `exp-150`–`154`, `157`–`159`. `utils/view_token_ops.py` for view-token eval paths.

### Installation

```bash
conda create -n self_forcing_games python=3.10 -y
conda activate self_forcing_games
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

Pretrained weights (must be downloaded once):

```bash
# Wan2.1 T2V-1.3B base
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir wan_models/Wan2.1-T2V-1.3B --local-dir-use-symlinks False
```

Melting Pot (only needed if you want to (re)generate the dataset; not needed for training from
the pre-built LMDBs):

```bash
pip install dm-meltingpot
```

### Quickstart: train the headline model (exp-139)

1. **Get the dataset.** Either generate it locally or download the pre-built LMDBs. See [`docs/DATASET.md`](docs/DATASET.md).
2. **Pretrain** the video-only base (`exp-138`) on the 46-game dataset, or download our released checkpoint.
3. **Fine-tune** with coord-token diffusion:

```bash
torchrun --nproc_per_node=1 --max_restarts=0 \
  train.py --config_path configs/exp-139-multi-game-coord-512.yaml
# or
bash scripts/train/run_139.sh 1
```

Per-config recipes and expected compute are in [`docs/TRAINING.md`](docs/TRAINING.md).

### Inference

```bash
python scripts/inference/inference_all_games_coord.py \
  --config_path configs/exp-139-multi-game-coord-512.yaml \
  --checkpoint_path checkpoints/exp-139-multi-game-coord-512/checkpoint_model_XXXXX/model.pt
```

This produces per-game comparison GIFs (ground-truth left, generated right with coord markers).

### Further reading

- [`docs/METHOD.md`](docs/METHOD.md) — architecture (coord tokens, spatial RoPE, masks)
- [`docs/DATASET.md`](docs/DATASET.md) — LMDB schema and dataset creation
- [`docs/TRAINING.md`](docs/TRAINING.md) — configs and compute
- [`docs/ABLATIONS.md`](docs/ABLATIONS.md) — ablation configs 141–145

## Acknowledgements

This code is forked from [Self-Forcing](https://github.com/guandeh17/Self-Forcing)
(Huang, Li, He, Zhou, Shechtman, 2025) and uses the Alibaba
[Wan2.1](https://github.com/Wan-Video/Wan2.1) T2V DiT. Datasets are rendered from
[DeepMind Melting Pot 2.0](https://github.com/google-deepmind/meltingpot).

## Citation

```bibtex
@article{pondaven2026actionparty,
      title={ActionParty: Multi-Subject Action Binding in Generative Video Games},
      author={Alexander Pondaven and Ziyi Wu and Igor Gilitschenski and Philip Torr and Sergey Tulyakov and Fabio Pizzati and Aliaksandr Siarohin},
      journal={arXiv preprint arXiv:2604.02330},
      year={2026},
}
```

## License

See [`LICENSE`](LICENSE): Snap Inc. sample-code terms (non-commercial research), plus
attribution and Apache License 2.0 text for Self-Forcing, Wan2.1, and Melting Pot portions.
