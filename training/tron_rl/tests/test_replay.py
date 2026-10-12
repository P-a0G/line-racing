import numpy as np

from tron_rl.replay import SumTree, Trajectory, ReplayBuffer
from tron_rl.obs import NUM_CHANNELS, HEIGHT, WIDTH


def test_sum_tree_basic():
    tree = SumTree(8)
    for i, v in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
        tree.set(i, v)
    assert tree.total() == 36
    assert tree.sample(0.5) == 0   # falls in leaf 0's [0,1) range
    assert tree.sample(2.5) == 1   # leaf 1's [1,3) range
    assert tree.sample(35.9) == 7  # last leaf


def make_trajectory(rewards):
    traj = Trajectory()
    grid = np.zeros((NUM_CHANNELS, HEIGHT, WIDTH), dtype=np.float32)
    legal = np.array([True, True, True, True])
    for _ in range(len(rewards) + 1):
        traj.append_observation(grid, legal)
    for a, r in enumerate(rewards):
        traj.append_action(a % 4, r)
    return traj


def test_trajectory_length_and_shapes():
    traj = make_trajectory([0.1, 0.2, 0.3])
    assert traj.length == 3
    assert len(traj.grids) == 4
    assert len(traj.legal) == 4


def test_n_step_return_and_terminal_flag_short_trajectory():
    # L=5 actions, n_step=3, gamma=1.0: windows truncate at the true episode
    # end (L), never bootstrapping past it -- the adaptation this buffer
    # exists for, since episodes no longer share one fixed MAX_TURNS length.
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    traj = make_trajectory(rewards)
    buf = ReplayBuffer(capacity_transitions=16, n_step=3, gamma=1.0)
    buf.add_trajectory(traj)

    assert len(buf) == 5
    expected_returns = [
        sum(rewards[0:3]),  # t=0: window [0,3)
        sum(rewards[1:4]),  # t=1: window [1,4)
        sum(rewards[2:5]),  # t=2: window [2,5) -- reaches the end, terminal
        sum(rewards[3:5]),  # t=3: window truncated at L=5, terminal
        sum(rewards[4:5]),  # t=4: window truncated at L=5, terminal
    ]
    expected_terminal = [False, False, True, True, True]
    np.testing.assert_allclose(buf.n_step_return[:5], expected_returns)
    np.testing.assert_array_equal(buf.terminal[:5], expected_terminal)
    # bootstrap_discount = gamma ** window_length; gamma=1.0 makes every one 1.0
    np.testing.assert_allclose(buf.bootstrap_discount[:5], [1.0] * 5)


def test_circular_wraparound_caps_size_at_capacity():
    buf = ReplayBuffer(capacity_transitions=10, n_step=2, gamma=1.0)
    for _ in range(5):
        buf.add_trajectory(make_trajectory([1.0, 1.0, 1.0]))  # 3 transitions each -> 15 total > capacity
    assert len(buf) == 10  # capped, not 15
    assert buf.write_pos == 5  # 15 writes into a 10-slot ring end up 5 past the wrap


def test_sample_shapes_and_priority_update():
    traj = make_trajectory([1.0, 2.0, 3.0, 4.0])
    buf = ReplayBuffer(capacity_transitions=16, n_step=3, gamma=0.99)
    buf.add_trajectory(traj)

    rng = np.random.default_rng(0)
    batch = buf.sample(batch_size=8, rng=rng)
    assert batch["grid_t"].shape == (8, NUM_CHANNELS, HEIGHT, WIDTH)
    assert batch["grid_t"].dtype == np.float32
    assert batch["legal_t"].shape == (8, 4)
    assert batch["action"].shape == (8,)
    assert batch["weights"].shape == (8,)
    assert np.isfinite(batch["grid_t"]).all()
    assert batch["weights"].max() <= 1.0 + 1e-6  # IS weights are normalized by their own max

    buf.update_priorities(batch["indices"], td_errors=np.full(8, 10.0))
    assert buf.max_priority > 1.0  # grew from the default after a large TD error


def test_nbytes_is_positive_and_matches_backing_arrays():
    buf = ReplayBuffer(capacity_transitions=100, n_step=3, gamma=1.0)
    assert buf.nbytes() > 0
    assert buf.nbytes() == sum(a.nbytes for a in (
        buf.grid_t, buf.legal_t, buf.action, buf.grid_tn, buf.legal_tn,
        buf.n_step_return, buf.bootstrap_discount, buf.terminal))
