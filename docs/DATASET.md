# Dataset Guide

All training configs in this repository consume pre-encoded LMDBs. The dataset
for the headline exp-139 run is ``melting_pot_all_games_512_2k`` (46
games × ~2000 samples each, VAE-encoded at 512×512, ≈113 GB on disk).

## Two ways to get the dataset

### Option 1 — download our pre-built LMDB

Not available yet. Until it is released, regenerate the dataset with Option 2.

The expected layout under ``datasets/melting_pot_all_games_512_2k`` is one
subfolder per game, each containing ``shard_0.lmdb/{data.mdb,lock.mdb}``.

### Option 2 — regenerate from Melting Pot

You will need DeepMind Melting Pot installed and a GPU to VAE-encode the
videos on the fly:

```bash
pip install dm-meltingpot
```

Then:

```bash
python scripts/dataset/create_all_games_dataset.py \
    --canvas_size 512 \
    --samples_per_game 2000 \
    --output_path datasets/melting_pot_all_games_512_2k
```

Run it from the repository root (the Wan VAE is loaded from ``wan_models/``).

Notes:
* ``--canvas_size 512`` uses the mini-map variants defined in
  ``scripts/dataset/mini_maps.py`` (each game padded / cropped to a 16×16
  tile grid at 4x scale = 512×512 pixels, 32 px sprites).
* ``--canvas_size 1024`` uses the full-size Melting Pot maps at 4x scale.
* ``--resume`` skips games that already have an LMDB on disk.
* Use ``--gpu N`` to pin to a specific GPU (sets ``CUDA_VISIBLE_DEVICES``).

Two auxiliary scripts are provided for the ablation sets:

* ``scripts/dataset/create_melting_pot_coins_original_actions.py`` — the
  coins-only dataset with the 7-action native Melting Pot action space,
  consumed by exp-134/135/141/142/143/145.
* ``scripts/dataset/create_random_blob_position_dataset_2subj.py`` — the
  2-subject synthetic shapes dataset used by exp-131.

### Downscaled smoke test

For a quick smoke test, pass e.g. ``--games coins --samples_per_game 10`` to
``create_all_games_dataset.py`` and train with a 1-game subset. Update the
config's ``data_path`` to point at that LMDB. When running
``scripts/inference/inference_all_games_coord.py`` on such a small set, pass
``--sample_offset 0`` (the default offset of 50 skips games with fewer samples).

## LMDB schema

Each game is a separate LMDB consumed by
:class:`utils.dataset.MultiGameLMDBDataset`. Sample ``i`` is stored as one
pickled array per field, under keys ``<field>_<i>_data``:

| field             | shape                     | dtype    | description                                             |
|-------------------|---------------------------|----------|---------------------------------------------------------|
| ``video_latents`` | ``[T_lat, 16, H, W]``     | float32  | VAE-encoded 17-frame clip (T_lat=5 for 17 pixel frames) |
| ``actions``       | ``[S, T_action]``         | int64    | Discrete per-player actions                             |
| ``frame_coords``  | ``[T_lat, S, 2]``         | float32  | ``(x, y)`` pixel coords on the canvas (0..canvas_size)   |
| ``orientations``  | ``[T_lat, S]``            | int64    | 0=N,1=E,2=S,3=W (tracked for relative-action games)     |
| ``initial_positions`` | ``[S, 2]``            | float32  | ``(x, y)`` at the first frame of the clip                |
| ``prompts``       | string                    | bytes    | Game slug (e.g. ``coins``)                              |

Per-game metadata is stored once per shard: ``num_samples``, ``num_subjects``
(actual S for this game, padded up to ``num_subjects_total`` at load time),
``num_actions``, ``game_name``, ``image_size`` and ``video_latents_shape``.

At load time, :class:`MultiGameLMDBDataset`:

1. Maps the game slug to a T5-embedded description pulled from
   :mod:`utils.game_descriptions` (cached once per training run).
2. Pads ``actions`` and ``frame_coords`` to ``num_subjects_total`` subjects,
   tracking a ``subject_mask`` so padded players do not contribute to loss.
3. Converts ``frame_coords`` from pixel units to normalized ``[0, 1]`` coords.

## List of games

All 46 mini-map games (512×512) are listed in
``scripts/dataset/create_all_games_dataset.py::GAMES_1024`` plus ``MINI_GAMES_512``
in ``scripts/dataset/mini_maps.py``. Visual descriptions used by the T5 text
encoder live in :mod:`utils.game_descriptions`.
