import numpy as np
import pytest
import torch

from tron_rl.arena import play_match, resolve_policy, _heuristic_policy, _random_policy
from tron_rl.learner import Learner
from tron_rl.train import _save_checkpoint
from tron_rl.opponents import SnapshotPool


def test_heuristic_beats_random_decisively():
    rng = np.random.default_rng(0)
    wins, mean_turns = play_match([_heuristic_policy, _random_policy(rng)], num_games=20, seed=0)
    assert sum(wins) == 20  # no draws possible, per Engine.hpp
    assert wins[0] > wins[1]
    assert mean_turns > 0


def test_three_player_heuristic_mirror_match_sums_to_games_played():
    wins, _ = play_match([_heuristic_policy, _heuristic_policy, _heuristic_policy], num_games=15, seed=1)
    assert sum(wins) == 15
    assert len(wins) == 3


def test_resolve_policy_unknown_spec_raises():
    with pytest.raises(ValueError):
        resolve_policy("not-a-real-policy")


def test_resolve_policy_heuristic_and_random():
    assert resolve_policy("heuristic") is _heuristic_policy
    policy = resolve_policy("random", np.random.default_rng(0))
    assert callable(policy)


def test_checkpoint_policy_loads_and_plays(tmp_path):
    torch.manual_seed(0)
    learner = Learner(net_kwargs=dict(channels=8, blocks=2, gpb_blocks=(0,)), num_taus=4)
    pool = SnapshotPool()
    pool.add(learner.online.state_dict())
    ckpt_path = tmp_path / "checkpoint.pt"
    _save_checkpoint(ckpt_path, learner, pool, it=0)

    net_policy = resolve_policy(f"checkpoint:{ckpt_path}")
    snap_policy = resolve_policy(f"snapshot:{ckpt_path}:0")
    wins, mean_turns = play_match([net_policy, snap_policy], num_games=4, seed=2)
    assert sum(wins) == 4
    assert mean_turns > 0


def test_rotation_attributes_wins_to_policy_not_seat():
    # Run enough games that the strong policy's seat position cycles through
    # every slot of a 3-player match; its win total should still dominate,
    # confirming wins[] tracks the POLICY, not whichever seat index it
    # happened to occupy in a given game.
    rng = np.random.default_rng(3)
    wins, _ = play_match([_heuristic_policy, _random_policy(rng), _random_policy(rng)], num_games=12, seed=3)
    assert wins[0] > wins[1] + wins[2]
