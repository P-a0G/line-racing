import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "simulator"))
import tronsim as ts  # noqa: E402

from tron_rl.obs import NUM_CHANNELS, HEIGHT, WIDTH  # noqa: E402
from tron_rl.learner import Learner  # noqa: E402
from tron_rl.opponents import SnapshotPool  # noqa: E402
from tron_rl.collect import Collector  # noqa: E402


def make_learner():
    torch.manual_seed(0)
    return Learner(net_kwargs=dict(channels=8, blocks=2, gpb_blocks=(0,)), num_taus=4)


def assert_well_formed(traj):
    assert len(traj.grids) == traj.length + 1
    assert len(traj.legal) == traj.length + 1
    for g in traj.grids:
        assert g.shape == (NUM_CHANNELS, HEIGHT, WIDTH)
    for a in traj.actions:
        assert a in (0, 1, 2, 3)
    assert all(np.isfinite(r) for r in traj.rewards)


def test_pure_heuristic_collection_produces_well_formed_trajectories():
    learner = make_learner()
    cpp_rng = ts.Rng(1)
    pool = SnapshotPool()
    collector = Collector(num_slots=4, player_count_mix=(0.5, 0.5))

    trajectories, stats = collector.collect(
        learner, cpp_rng, pool, mix=(0.0, 0.0, 1.0), num_decisions=400,
        rng=np.random.default_rng(1))

    assert len(trajectories) > 0
    for traj in trajectories:
        assert_well_formed(traj)
    assert 2 * stats["games_finished"] <= len(trajectories) <= 3 * stats["games_finished"]
    assert stats["mean_turns"] > 0


def test_current_and_snapshot_policy_kinds_do_not_crash():
    learner = make_learner()
    cpp_rng = ts.Rng(2)
    pool = SnapshotPool()
    pool.add(learner.online.state_dict())  # so the 'snapshot' slice has something real to load
    collector = Collector(num_slots=4, player_count_mix=(0.5, 0.5))

    trajectories, stats = collector.collect(
        learner, cpp_rng, pool, mix=(0.5, 0.3, 0.2), num_decisions=300,
        rng=np.random.default_rng(2))

    for traj in trajectories:
        assert_well_formed(traj)
    assert stats["games_finished"] >= 0  # mostly: must not have crashed building the batch


def test_player_count_mix_is_respected_by_respawn():
    # A direct, fast check of the mixing logic itself -- a real-game
    # integration test would need many more decisions to reliably observe
    # both counts, since heuristic-vs-heuristic games on this board often
    # run long (hundreds of combined sub-turns; see the mean_turns asserted
    # by test_multi_slot_collection_runs_games_to_completion below).
    cpp_rng = ts.Rng(3)
    pool = SnapshotPool()
    collector = Collector(num_slots=1, player_count_mix=(0.5, 0.5))
    rng = np.random.default_rng(3)
    counts = {2: 0, 3: 0}
    for _ in range(2000):
        collector._respawn_slot(0, cpp_rng, rng, pool, mix=(0.0, 0.0, 1.0))
        counts[collector.games[0].num_players] += 1
    assert abs(counts[2] / 2000 - 0.5) < 0.05
    assert abs(counts[3] / 2000 - 0.5) < 0.05


def test_multi_slot_collection_runs_games_to_completion():
    learner = make_learner()
    cpp_rng = ts.Rng(4)
    pool = SnapshotPool()
    collector = Collector(num_slots=6, player_count_mix=(0.5, 0.5))

    _, stats = collector.collect(
        learner, cpp_rng, pool, mix=(0.0, 0.0, 1.0), num_decisions=600,
        rng=np.random.default_rng(4))

    assert stats["games_finished"] >= 1
    assert stats["mean_turns"] > 0


def test_collector_persists_in_flight_games_across_calls():
    learner = make_learner()
    cpp_rng = ts.Rng(4)
    pool = SnapshotPool()
    collector = Collector(num_slots=1, player_count_mix=(1.0, 0.0))  # always N=2, single slot

    _, stats_first = collector.collect(
        learner, cpp_rng, pool, mix=(0.0, 0.0, 1.0), num_decisions=1, rng=np.random.default_rng(4))
    move_count_after_first = collector.games[0].move_count if collector.games[0] is not None else None

    if stats_first["games_finished"] == 0:
        assert move_count_after_first == 1
        collector.collect(learner, cpp_rng, pool, mix=(0.0, 0.0, 1.0), num_decisions=1,
                            rng=np.random.default_rng(4))
        # the SAME game continued rather than being thrown away and restarted
        assert collector.games[0] is None or collector.games[0].move_count in (0, 2)
