import numpy as np

from tron_rl.opponents import SnapshotPool, sample_policy_kind


def test_pure_current_mix_always_returns_current():
    rng = np.random.default_rng(0)
    pool = SnapshotPool()
    for _ in range(50):
        kind, payload = sample_policy_kind(rng, pool, mix=(1.0, 0.0, 0.0))
        assert kind == "current"
        assert payload is None


def test_snapshot_falls_back_to_current_when_pool_empty():
    rng = np.random.default_rng(1)
    pool = SnapshotPool()
    for _ in range(50):
        kind, _ = sample_policy_kind(rng, pool, mix=(0.5, 0.5, 0.0))
        assert kind != "snapshot"  # nothing to sample yet, falls back to current


def test_heuristic_mix_returns_a_callable():
    rng = np.random.default_rng(2)
    pool = SnapshotPool()
    kind, payload = sample_policy_kind(rng, pool, mix=(0.0, 0.0, 1.0))
    assert kind == "heuristic"
    assert callable(payload)


def test_mix_proportions_are_roughly_respected():
    rng = np.random.default_rng(3)
    pool = SnapshotPool()
    pool.add({"dummy": 1})
    counts = {"current": 0, "snapshot": 0, "heuristic": 0}
    n = 20_000
    for _ in range(n):
        kind, _ = sample_policy_kind(rng, pool, mix=(0.5, 0.3, 0.2))
        counts[kind] += 1
    assert abs(counts["current"] / n - 0.5) < 0.02
    assert abs(counts["snapshot"] / n - 0.3) < 0.02
    assert abs(counts["heuristic"] / n - 0.2) < 0.02


def test_snapshot_pool_fifo_eviction_and_deepcopy_isolation():
    pool = SnapshotPool(max_size=2)
    d1 = {"v": 1}
    pool.add(d1)
    pool.add({"v": 2})
    pool.add({"v": 3})  # evicts d1's slot
    assert len(pool) == 2
    values = {s["v"] for s in pool.snapshots}
    assert values == {2, 3}

    d4 = {"v": 4}
    pool.add(d4)
    d4["v"] = 999  # mutating the original after add() must not affect the stored copy
    assert pool.snapshots[-1]["v"] == 4
