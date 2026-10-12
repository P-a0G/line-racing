"""Self-play opponent mixing: each seat in a collected episode is
independently assigned a policy kind -- current network (NoisyNet
exploration on), a frozen past snapshot (small FIFO pool), or the C++
flood-fill heuristic -- rather than every episode being pure self-play. This
is the same anti-exploit anchor summer_challenge_2026/training/rlx/
opponents.py documents against the classic self-play failure mode where a
policy overfits to beating its own recent history without actually getting
stronger; ported here near-verbatim (the only difference is a single
heuristic instead of two, since line-racing's simulator only has one).

Because this is off-policy Q-learning, every turn's transition is valid to
learn from regardless of which policy chose the action -- collect.py records
every seat every episode no matter the mix; the mix controls the
state-action distribution training sees, not which transitions are usable.
This is already seat-count-agnostic (called once per seat); collect.py is
what generalizes the seat *count* to 2 or 3.
"""
import copy
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "simulator"))
import tronsim as ts  # noqa: E402

HEURISTICS = (ts.heuristic_action,)


class SnapshotPool:
    def __init__(self, max_size=20):
        self.max_size = max_size
        self.snapshots = deque(maxlen=max_size)

    def add(self, state_dict):
        self.snapshots.append(copy.deepcopy(state_dict))

    def sample(self, rng):
        if not self.snapshots:
            return None
        return self.snapshots[int(rng.integers(len(self.snapshots)))]

    def __len__(self):
        return len(self.snapshots)


def sample_policy_kind(rng, snapshot_pool, mix=(0.5, 0.3, 0.2)):
    """Returns (kind, payload): kind in {'current','snapshot','heuristic'};
    payload is None for 'current', a state_dict for 'snapshot', or a tronsim
    policy callable for 'heuristic'. Falls back to 'current' for the
    snapshot slice until the pool holds anything."""
    p_current, p_snapshot, p_heuristic = mix
    if len(snapshot_pool) == 0:
        p_current, p_snapshot = p_current + p_snapshot, 0.0
    r = rng.uniform(0, p_current + p_snapshot + p_heuristic)
    if r < p_current:
        return "current", None
    if r < p_current + p_snapshot:
        return "snapshot", snapshot_pool.sample(rng)
    return "heuristic", HEURISTICS[int(rng.integers(len(HEURISTICS)))]
