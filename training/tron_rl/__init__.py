"""The ported BTR/Indrill-style training pipeline for line-racing (see
C:/Users/33783/.claude/plans/bright-baking-flask.md).

Puts submission/tron_infer/ on sys.path so every submodule here can just
`import features_core` directly -- that directory is the single source of
truth shared with the real CG submission, and this is the one place the
path glue lives.
"""
import sys
from pathlib import Path

_SUBMISSION_INFER = Path(__file__).resolve().parent.parent.parent / "submission" / "tron_infer"
if str(_SUBMISSION_INFER) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION_INFER))
