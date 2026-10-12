"""Training-side entrypoint to the feature engineering module.

The actual implementation lives in `submission/tron_infer/features_core.py`
and is imported here unmodified -- that file is the single source of truth
shared with the real CG submission (see build_submission.py), so there is
exactly one place the feature math is written, avoiding training-vs-
submission drift.
"""
import numpy as np

import tron_rl  # noqa: F401  (side effect: puts submission/tron_infer on sys.path)
from features_core import (  # noqa: E402,F401
    WIDTH, HEIGHT, MAX_OPPONENTS, CHANNEL_NAMES, NUM_CHANNELS,
    build_observation,
)


def state_to_arrays(state):
    """Converts a tronsim.GameState (the pybind11 bridge's flat arrays) into
    the (grid_owner, heads, alive) representation build_observation() and
    reward.potential() expect: grid_owner is (HEIGHT, WIDTH) int8 with -1
    free else the owning player id; heads is a list of (x, y) or None per
    player id; alive is a list of bool per player id. Only obs.py (not
    features_core.py) is allowed to know tronsim exists -- see tron_rl's
    module docstring."""
    grid_owner = np.asarray(state.cell_owner, dtype=np.int8).reshape(HEIGHT, WIDTH)
    alive = list(state.alive[:state.num_players])
    heads = []
    for p in range(state.num_players):
        if alive[p]:
            cell = state.head_cell[p]
            heads.append((cell % WIDTH, cell // WIDTH))
        else:
            heads.append(None)
    return grid_owner, heads, alive
