# Ablations (exp-141 through 145) and text-action baselines (144, 146)

This page explains what each "ablation" config toggles and what it tells us
about the coord-token method.

The core coord-token method is **exp-134** (coins, 7-action) and
**exp-139** (46-game Melting Pot). All ablations in the 141–145 range
start from the **same** video-only pretrain
(``checkpoints/exp-135-base-wan-pretrain-original-actions/checkpoint_model_007000/model.pt``)
and differ only in which piece of the method is removed.

## The headline components being ablated

| Component                     | Where it lives                                                     | Default in 134 | Default in 139 |
|-------------------------------|--------------------------------------------------------------------|----------------|----------------|
| Coord tokens in the sequence  | ``coord_token_mode: true``                                         | yes            | yes            |
| Joint coord-video diffusion   | ``coord_diffusion: true``                                          | yes            | yes            |
| Subject isolation in self-attn | ``subject_isolation_self_attn: true`` + FlexAttention block mask | yes            | yes            |
| RoPE on coord tokens in self-attn | ``coord_token_rope_in_self_attn: true``                         | yes            | yes            |
| Spatial-RoPE cross-attention  | ``cross_attn_type: masked_position_cross_attn``                     | yes            | yes            |
| Per-subject masking in cross-attn | ``cross_attn_subject_isolation: true``                          | yes            | yes            |

## Ablation configs

### exp-141 no-self-attn-isolation
Drops the subject-isolation mask in self-attention (coord tokens of subject A
can now attend to coord tokens of subject B).
**Tests:** do we need to keep subjects from leaking into each other's coord
tokens to learn disentangled representations?

### exp-142 no-cross-attn-isolation
Drops per-subject masking in the spatial cross-attention (every action token
is visible to every video query).
**Tests:** is it enough to give the model spatial RoPE, or do we also need
to structurally prevent players from seeing each others' action tokens?

### exp-143 per-frame-cross-attn-coord
Switches cross-attention from per-subject (S masks × T frames) to per-frame
only (one mask × T frames). Action tokens are attended to by all subjects in
a given frame.
**Tests:** is subject-level masking overkill, or does frame-level suffice?

### exp-145 no-self-attn-rope-coord
Keeps all masks but removes RoPE from the coord-token stream inside
self-attention (coord tokens become "bag of coordinates" with no explicit
spatial frequency encoding).
**Tests:** how much of the benefit is from RoPE vs. from the coord-token
architecture itself?

### exp-144 text-action-history-512 (baseline, not ablation)
A *strong* baseline: no coord tokens, no spatial cross-attention. The
history of actions (up to the current frame) is rendered as a string, fed
through T5 alongside the game description, and used as the only action
conditioning. Same pretrain as 139 (``exp-138``).
**Tests:** how far can we get with just a bigger text prompt?

### exp-146 text-action-20actions-512
Same as 144 but trained on the 25-action variant of the dataset (covers
full Melting Pot action space including ``TURN_LEFT/RIGHT`` and
``FIRE_ZAP``). Starts from the 144 checkpoint.

### exp-147 coord-20actions-512
The coord-token counterpart to 146 — the 25-action variant of the headline
method. Starts from the 139 checkpoint (so we compare 146 vs. 147 on matched
compute).

## How to read the results

Expected ordering (in-distribution PSNR / action-following metric on
held-out seeds):

```
 exp-139 / 147           (full coord-token method)
 > exp-145 / 141 / 142   (mild drops from individual components)
 > exp-143               (per-frame is noticeably worse)
 >> exp-144 / 146        (text-action baseline)
```

Use the 141–145 group to attribute quality to specific design choices, and
139 vs. 144 (or 147 vs. 146 on matched action-space) for the overall "coord
tokens vs. text-only" headline comparison.
