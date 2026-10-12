"""Training-side entrypoint to the feature engineering module.

The actual implementation lives in `submission/tron_infer/features_core.py`
and is imported here unmodified -- that file is the single source of truth
shared with the real CG submission (see build_submission.py), so there is
exactly one place the feature math is written, avoiding training-vs-
submission drift.
"""
import tron_rl  # noqa: F401  (side effect: puts submission/tron_infer on sys.path)
from features_core import (  # noqa: E402,F401
    WIDTH, HEIGHT, MAX_OPPONENTS, CHANNEL_NAMES, NUM_CHANNELS,
    build_observation,
)
