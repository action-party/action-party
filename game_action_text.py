"""Text encoding of per-player actions for the text-action baselines (exp-144, exp-146).

The original implementation is not included in this release. ``trainer/diffusion.py``
imports these two functions when ``text_action_conditioning: true``; the returned
string is appended to the game description and encoded with T5.

Expected output (from ``configs/exp-144-text-action-history-512.yaml``):

    player 1 starts at (3,4). player 2 starts at (7,2).
    In frame 1, player 1 moves forward. player 2 turns right.
    In frame 2, player 1 strafes left. player 2 stands still.
"""

_MISSING = (
    "game_action_text is not included in this release, so the text-action "
    "baselines (exp-144, exp-146) cannot be trained yet. See this module's "
    "docstring for the expected interface."
)


def build_action_text(game_name, subj_actions, subj_positions, num_real_subjects, compact=False):
    """Describe one frame: each real player's position and action.

    Args:
        game_name: Melting Pot substrate name, e.g. ``"clean_up"``.
        subj_actions: [S] action indices for that frame (padded to S players).
        subj_positions: [S, 2] ``(x, y)`` coordinates for that frame, normalized to [0, 1].
        num_real_subjects: number of real (unpadded) players.
        compact: shorter phrasing (``compact_action_text`` in the config).
    """
    raise NotImplementedError(_MISSING)


def build_multi_frame_action_text(game_name, actions, frame_coords, num_real_subjects,
                                  num_frames, compact=False):
    """Describe initial positions, then each player's action for frames 1..num_frames.

    Args:
        game_name: Melting Pot substrate name, e.g. ``"clean_up"``.
        actions: [S, num_steps] action indices (padded to S players).
        frame_coords: [T, S, 2] ``(x, y)`` coordinates, normalized to [0, 1]. Only
            frame 0 is described, so later positions are not leaked to the model.
        num_real_subjects: number of real (unpadded) players.
        num_frames: number of context frames whose actions are described.
        compact: shorter phrasing (``compact_action_text`` in the config).
    """
    raise NotImplementedError(_MISSING)
