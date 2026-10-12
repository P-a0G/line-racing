"""Prioritized experience replay with n-step returns, for the BTR-style
learner (learner.py).

Unlike summer_challenge_2026's replay.py -- whose episodes always run
exactly MAX_TURNS turns, so trajectories can be preallocated fixed-size
arrays indexed by (slot, offset) in a shared sum-tree -- a line-racing
episode's length varies per game (it ends the moment a player is
eliminated). So this buffer stores individual TRANSITIONS directly in flat
circular arrays: the n-step return and the bootstrap-target observation are
both precomputed once, when a finished trajectory is added, directly from
that trajectory's own rewards/observations. This duplicates the bootstrap
observation's storage (summer_challenge_2026 shares it lazily via a kept-
alive Trajectory object instead), trading a bit of memory for a much
simpler design given episodes no longer share one fixed length worth
indexing into.
"""
import numpy as np

from tron_rl.obs import NUM_CHANNELS, HEIGHT, WIDTH

NUM_ACTIONS = 4  # UP, DOWN, LEFT, RIGHT -- see simulator/Engine.hpp / net.py
GRID_SCALE = 4.0 / 127.0  # every features_core.py channel is roughly in [-1,1]; this leaves headroom


def quantize_grid(grid_f32):
    return np.clip(np.round(grid_f32 / GRID_SCALE), -127, 127).astype(np.int8)


def dequantize_grid(grid_i8):
    return grid_i8.astype(np.float32) * GRID_SCALE


class SumTree:
    """Array-based binary heap sum-tree over `capacity` leaves: O(log n)
    update and prefix-sum sampling, O(1) total(). Copied from
    summer_challenge_2026/training/rlx/replay.py -- fully generic, no
    adaptation needed (works for any capacity, not just powers of 2: every
    internal index i in [1,capacity) has both children 2i/2i+1 within the
    2*capacity-sized array by construction)."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    def set(self, leaf_idx, value):
        i = leaf_idx + self.capacity
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i //= 2

    def total(self):
        return self.tree[1]

    def get_leaf(self, leaf_idx):
        return self.tree[leaf_idx + self.capacity]

    def sample(self, value):
        """Returns the leaf index whose cumulative range contains `value`
        (0 <= value < total())."""
        i = 1
        while i < self.capacity:
            left = 2 * i
            if value < self.tree[left]:
                i = left
            else:
                value -= self.tree[left]
                i = left + 1
        return i - self.capacity


class Trajectory:
    """One (episode, seat)'s data, appended to turn by turn as the game is
    played -- a plain growable list rather than summer_challenge_2026's
    preallocated fixed-size arrays, since an episode's length here isn't
    known ahead of time. `grids`/`legal` have length L+1 (observations
    0..L inclusive); `actions`/`rewards` have length L (turn i's action
    produced the transition from observation i to i+1)."""

    def __init__(self):
        self.grids = []
        self.legal = []
        self.actions = []
        self.rewards = []

    def append_observation(self, grid_f32, legal_mask):
        self.grids.append(quantize_grid(grid_f32))
        self.legal.append(np.asarray(legal_mask, dtype=bool))

    def append_action(self, action, reward):
        self.actions.append(int(action))
        self.rewards.append(float(reward))

    @property
    def length(self):
        return len(self.actions)


class ReplayBuffer:
    def __init__(self, capacity_transitions=100_000, n_step=3, gamma=1.0,
                 alpha=0.2, beta=0.45, eps=1e-3):
        self.capacity = capacity_transitions
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.grid_t = np.zeros((capacity_transitions, NUM_CHANNELS, HEIGHT, WIDTH), dtype=np.int8)
        self.legal_t = np.zeros((capacity_transitions, NUM_ACTIONS), dtype=bool)
        self.action = np.zeros((capacity_transitions,), dtype=np.int8)
        self.grid_tn = np.zeros((capacity_transitions, NUM_CHANNELS, HEIGHT, WIDTH), dtype=np.int8)
        self.legal_tn = np.zeros((capacity_transitions, NUM_ACTIONS), dtype=bool)
        self.n_step_return = np.zeros((capacity_transitions,), dtype=np.float32)
        self.bootstrap_discount = np.zeros((capacity_transitions,), dtype=np.float32)
        self.terminal = np.zeros((capacity_transitions,), dtype=bool)

        self.tree = SumTree(capacity_transitions)
        self.write_pos = 0
        self.size = 0
        self.max_priority = 1.0

    def nbytes(self):
        """Actual resident size of the buffer's backing arrays -- call this
        right after construction and sanity-check it against available RAM;
        summer_challenge_2026 shipped a hardcoded trajectory capacity that
        silently worked out to 10.4GB before anyone checked."""
        arrays = (self.grid_t, self.legal_t, self.action, self.grid_tn, self.legal_tn,
                  self.n_step_return, self.bootstrap_discount, self.terminal)
        return sum(a.nbytes for a in arrays)

    def __len__(self):
        return self.size

    def add_trajectory(self, traj):
        L = traj.length
        for t in range(L):
            end = min(t + self.n_step, L)
            discounts = self.gamma ** np.arange(end - t)
            ret = float(np.dot(discounts, traj.rewards[t:end]))
            terminal = end == L
            bootstrap_discount = self.gamma ** (end - t)

            i = self.write_pos
            self.grid_t[i] = traj.grids[t]
            self.legal_t[i] = traj.legal[t]
            self.action[i] = traj.actions[t]
            self.grid_tn[i] = traj.grids[end]
            self.legal_tn[i] = traj.legal[end]
            self.n_step_return[i] = ret
            self.bootstrap_discount[i] = bootstrap_discount
            self.terminal[i] = terminal

            self.tree.set(i, self.max_priority ** self.alpha)
            self.write_pos = (self.write_pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def update_priorities(self, indices, td_errors):
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        self.max_priority = max(self.max_priority, float(priorities.max(initial=0.0)))
        for i, p in zip(indices, priorities):
            self.tree.set(int(i), float(p))

    def sample(self, batch_size, rng=None):
        rng = rng or np.random.default_rng()
        total = self.tree.total()
        indices = np.empty(batch_size, dtype=np.int64)
        segment = total / batch_size
        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            indices[i] = self.tree.sample(rng.uniform(lo, hi))

        probs = np.array([self.tree.get_leaf(idx) for idx in indices]) / total
        weights = (len(self) * probs) ** (-self.beta)
        weights /= weights.max()

        return {
            "indices": indices,
            "weights": weights.astype(np.float32),
            "grid_t": dequantize_grid(self.grid_t[indices]),
            "legal_t": self.legal_t[indices],
            "action": self.action[indices],
            "grid_tn": dequantize_grid(self.grid_tn[indices]),
            "legal_tn": self.legal_tn[indices],
            "n_step_return": self.n_step_return[indices],
            "bootstrap_discount": self.bootstrap_discount[indices],
            "terminal": self.terminal[indices],
        }
