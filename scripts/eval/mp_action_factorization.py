"""
Factor Melting Pot **joint** discrete action ids into independent softmax heads.

Each row in NPZ stores one integer ``ya`` indexing ``substrate.get_config(game).action_set``.
We map that joint id to four supervised targets:

- ``move``  — class index over sorted unique ``move`` values in the action set
- ``turn``  — class index over sorted unique ``turn`` values (typically ``{-1,0,1}``)
- ``fire_zap`` / ``fire_clean`` — 0/1 (2-class softmax each)

Used when ``--action_decomposition factored`` in ``train_per_game_action_tile_model.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class FactoredActionSpec:
    """Maps global joint action id ``orig_id`` (index into ``action_set``) to per-head class ids."""

    n_move: int
    n_turn: int
    n_zap: int
    n_clean: int
    move_values: tuple[int, ...]
    turn_values: tuple[int, ...]
    zap_values: tuple[int, ...]
    clean_values: tuple[int, ...]
    # orig_id -> (ym, yt, yz, yc) each in range for its head
    joint_to_factored: tuple[tuple[int, int, int, int], ...]

    def targets_for_orig(self, orig_id: int) -> tuple[int, int, int, int]:
        if orig_id < 0 or orig_id >= len(self.joint_to_factored):
            raise KeyError(f"orig_id {orig_id} out of range for factored spec (n_joint={len(self.joint_to_factored)})")
        return self.joint_to_factored[orig_id]


def load_action_set_dicts(game_name: str) -> tuple[dict[str, Any], ...]:
    from meltingpot import substrate as substrate_module

    cfg = substrate_module.get_config(str(game_name))
    aset = getattr(cfg, "action_set", None)
    if aset is None:
        raise ValueError(f"No action_set on config for game_name={game_name!r}")
    return tuple(dict(x) for x in aset)


def build_factored_spec(game_name: str) -> FactoredActionSpec:
    """Build spec from Melting Pot ``action_set`` ordering (joint id = index in tuple)."""
    aset = load_action_set_dicts(game_name)
    moves = sorted({int(d["move"]) for d in aset})
    turns = sorted({int(d["turn"]) for d in aset})
    zaps = sorted({int(d.get("fireZap", 0)) for d in aset})
    cleans = sorted({int(d.get("fireClean", 0)) for d in aset})
    mv = tuple(moves)
    tv = tuple(turns)
    zv = tuple(zaps)
    cv = tuple(cleans)
    j2f: list[tuple[int, int, int, int]] = []
    for d in aset:
        j2f.append(
            (
                mv.index(int(d["move"])),
                tv.index(int(d["turn"])),
                zv.index(int(d.get("fireZap", 0))),
                cv.index(int(d.get("fireClean", 0))),
            )
        )
    return FactoredActionSpec(
        n_move=len(mv),
        n_turn=len(tv),
        n_zap=len(zv),
        n_clean=len(cv),
        move_values=mv,
        turn_values=tv,
        zap_values=zv,
        clean_values=cv,
        joint_to_factored=tuple(j2f),
    )


def game_name_from_npz_path(path: str) -> str:
    import numpy as np

    with np.load(path, allow_pickle=True) as z:
        if "game_name" not in z.files:
            raise ValueError(f"{path!r} has no game_name array (needed for factored actions)")
        gn = z["game_name"]
        if hasattr(gn, "item"):
            gn = gn.item()
        s = str(gn)
        if s.startswith("b'") and s.endswith("'"):
            s = s[2:-1]
        return str(s)
