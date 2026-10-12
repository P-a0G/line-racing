import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission"))
import build_submission  # noqa: E402

from tron_rl.learner import Learner
from tron_rl.opponents import SnapshotPool
from tron_rl.train import _save_checkpoint
from tron_rl.export import export_checkpoint

REAL_SUBMISSION_DIR = Path(__file__).resolve().parents[3] / "submission"
WEIGHTS_BLOB_PATH = REAL_SUBMISSION_DIR / "tron_infer" / "weights_blob.py"


@pytest.fixture
def exported_weights_blob(tmp_path):
    """build_submission.build() reads from the real submission/tron_infer/
    directory (not configurable -- it's meant to be run as a standalone
    script against the real checked-out tree), so this generates a real
    weights_blob.py there for the test and restores whatever (if anything)
    was there before, rather than leaving a stray generated file behind."""
    torch.manual_seed(0)
    net_kwargs = dict(channels=8, blocks=2, gpb_blocks=(0,))
    learner = Learner(net_kwargs=net_kwargs, num_taus=4)
    pool = SnapshotPool()
    ckpt_path = tmp_path / "checkpoint.pt"
    _save_checkpoint(ckpt_path, learner, pool, it=0)

    backup = WEIGHTS_BLOB_PATH.read_text(encoding="utf-8") if WEIGHTS_BLOB_PATH.exists() else None
    export_checkpoint(ckpt_path, net_kwargs, WEIGHTS_BLOB_PATH)
    try:
        yield
    finally:
        if backup is not None:
            WEIGHTS_BLOB_PATH.write_text(backup, encoding="utf-8")
        else:
            WEIGHTS_BLOB_PATH.unlink(missing_ok=True)


def test_build_produces_a_valid_standalone_module(exported_weights_blob, tmp_path):
    out_path, size = build_submission.build(out_path=tmp_path / "submission.py")
    assert out_path.exists()
    assert 0 < size < build_submission.SAFE_CHAR_BUDGET  # this tiny test net should be nowhere close

    source = out_path.read_text(encoding="utf-8")
    code = compile(source, "<test-submission>", "exec")  # redundant with build()'s own check, kept explicit here

    namespace = {"__name__": "not_main"}  # prevents main()'s stdin loop from running on exec
    exec(code, namespace)

    # The concatenation must leave every piece independently callable, with
    # cross-module references (features_core's WIDTH/HEIGHT, forward(), the
    # weights_blob's load()) resolved into the one shared namespace.
    weights = namespace["load"]()
    grid = np.full((namespace["HEIGHT"], namespace["WIDTH"]), -1, dtype=np.int8)
    legal = np.array([True, True, True, True])
    obs = namespace["build_observation"](grid, [(5, 5), (10, 10)], [True, True], 0, 0)
    out = namespace["forward"](obs, legal, namespace["_MIDPOINT_TAUS"], weights)
    assert out["q"].shape == (len(namespace["_MIDPOINT_TAUS"]), 4)

    trails = {}
    _, heads, alive = namespace["parse_round"](2, ["5 5 5 5", "10 10 10 10"], trails)
    assert heads == [(5, 5), (10, 10)]
    assert alive == [True, True]


def test_build_fails_loudly_without_a_weights_blob(tmp_path):
    if WEIGHTS_BLOB_PATH.exists():
        pytest.skip("a real weights_blob.py already exists -- this test only covers the missing-file case")
    with pytest.raises(FileNotFoundError):
        build_submission.build(out_path=tmp_path / "submission.py")
