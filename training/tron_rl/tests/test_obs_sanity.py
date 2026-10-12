import numpy as np
from tron_rl.obs import build_observation, CHANNEL_NAMES, NUM_CHANNELS, WIDTH, HEIGHT
from features_core import _voronoi_ownership, _articulation_points


def empty_grid():
    return np.full((HEIGHT, WIDTH), -1, dtype=np.int8)


def test_shape_dtype_and_finiteness():
    grid = empty_grid()
    heads = [(5, 5), (20, 15)]
    grid[5, 5] = 0
    grid[15, 20] = 1
    obs = build_observation(grid, heads, alive=[True, True], my_id=0, turn=3)
    assert obs.shape == (NUM_CHANNELS, HEIGHT, WIDTH)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_two_player_opp2_channels_are_zeroed():
    grid = empty_grid()
    heads = [(5, 5), (20, 15)]
    grid[5, 5] = 0
    grid[15, 20] = 1
    obs = build_observation(grid, heads, alive=[True, True], my_id=0, turn=0)
    idx = {name: i for i, name in enumerate(CHANNEL_NAMES)}
    assert np.all(obs[idx["opp2_trail"]] == 0)
    assert np.all(obs[idx["opp2_head"]] == 0)
    assert np.all(obs[idx["opp2_alive"]] == 0)
    assert np.all(obs[idx["voronoi_opp2"]] == 0)
    # opp1 (the real, alive opponent) must look "present"
    assert np.all(obs[idx["opp1_alive"]] == 1.0)
    assert obs[idx["opp1_head"]].sum() == 1.0


def test_dead_opponent_looks_like_a_nonexistent_one():
    grid = empty_grid()
    heads = [(5, 5), (20, 15), (10, 10)]
    grid[5, 5] = 0
    grid[15, 20] = 1
    # player 2 eliminated: no head reported (None), alive=False
    obs_dead = build_observation(grid, [(5, 5), (20, 15), None], alive=[True, True, False], my_id=0, turn=10)
    obs_absent = build_observation(grid, [(5, 5), (20, 15)], alive=[True, True], my_id=0, turn=10)
    # both should encode "no second opponent" identically on every channel
    np.testing.assert_array_equal(obs_dead, obs_absent)


def test_my_head_is_one_hot():
    grid = empty_grid()
    grid[7, 12] = 0
    obs = build_observation(grid, heads=[(12, 7), (0, 0)], alive=[True, True], my_id=0, turn=0)
    idx = CHANNEL_NAMES.index("my_head")
    assert obs[idx].sum() == 1.0
    assert obs[idx][7, 12] == 1.0


def test_voronoi_symmetric_split_matches_engine_expectation():
    # Same scenario as simulator/test_voronoi.cpp's Test 1: heads on the same
    # row at opposite edges split the board 299/299 with ties to the lower id.
    grid = empty_grid()
    grid[10, 0] = 0
    grid[10, 29] = 1
    owner = _voronoi_ownership(grid, heads=[(0, 10), (29, 10)], alive=[True, True])
    assert (owner == 0).sum() == 15 * HEIGHT - 1
    assert (owner == 1).sum() == 15 * HEIGHT - 1
    assert owner[10, 14] == 0
    assert owner[10, 15] == 1


def test_articulation_point_flags_a_single_cell_bridge():
    # Two open rooms connected by exactly one bridge cell.
    free = np.ones((HEIGHT, WIDTH), dtype=bool)
    free[:, 10] = False      # wall column splitting the board...
    free[5, 10] = True       # ...except one bridge cell at (10, 5)
    art = _articulation_points(free)
    assert art[5, 10] == True
    # a cell deep inside one room, far from the bridge, must not be flagged
    assert art[2, 2] == False


def test_num_alive_and_turn_number_scale_with_inputs():
    grid = empty_grid()
    heads = [(5, 5), (20, 15), (1, 1)]
    for p, (x, y) in enumerate(heads):
        grid[y, x] = p
    obs_full = build_observation(grid, heads, alive=[True, True, True], my_id=0, turn=0)
    obs_one_dead = build_observation(grid, heads, alive=[True, True, False], my_id=0, turn=0)
    idx = CHANNEL_NAMES.index("num_alive")
    assert obs_full[idx][0, 0] > obs_one_dead[idx][0, 0]

    obs_late = build_observation(grid, heads, alive=[True, True, True], my_id=0, turn=150)
    idx_turn = CHANNEL_NAMES.index("turn_number")
    assert obs_late[idx_turn][0, 0] > obs_full[idx_turn][0, 0]
