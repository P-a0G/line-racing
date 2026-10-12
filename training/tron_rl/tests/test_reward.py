import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "simulator"))
import tronsim as ts  # noqa: E402

from tron_rl.obs import HEIGHT, WIDTH, state_to_arrays  # noqa: E402
from tron_rl.reward import potential, reward_from_potentials, terminal_bonus  # noqa: E402


def empty_grid():
    return np.full((HEIGHT, WIDTH), -1, dtype=np.int8)


def test_potential_symmetric_split_matches_voronoi_fractions():
    grid = empty_grid()
    grid[10, 0] = 0
    grid[10, 29] = 1
    heads = [(0, 10), (29, 10)]
    alive = [True, True]
    p0 = potential(grid, heads, alive, 0)
    p1 = potential(grid, heads, alive, 1)
    assert p0 == pytest.approx(299 / 598)
    assert p1 == pytest.approx(299 / 598)
    assert p0 + p1 == pytest.approx(1.0)  # zero-sum by construction


def test_eliminated_player_has_zero_potential():
    grid = empty_grid()
    grid[10, 0] = 0
    heads = [(0, 10), None]
    alive = [True, False]
    assert potential(grid, heads, alive, 1) == 0.0


def test_terminal_bonus_values():
    assert terminal_bonus(won=True, lost=False) == 1.0
    assert terminal_bonus(won=False, lost=True) == -1.0
    assert terminal_bonus(won=False, lost=False) == 0.0


def test_terminal_bonus_rejects_simultaneous_win_and_loss():
    with pytest.raises(AssertionError):
        terminal_bonus(won=True, lost=True)


def test_reward_from_potentials_is_a_plain_delta():
    assert reward_from_potentials(0.3, 0.45) == pytest.approx(0.15)
    assert reward_from_potentials(0.6, 0.6) == 0.0


def test_potential_stays_in_unit_range_over_a_real_playthrough():
    # Smoke-tests potential() against every state a real heuristic-vs-
    # heuristic game actually visits (not just hand-built boards): it must
    # never go out of [0,1] or produce a NaN, and a freshly spawned board
    # (symmetric in expectation, though any one random draw needn't be
    # exact) should start in a plausible range for two players splitting
    # the whole board, not pinned at an extreme.
    rng = ts.Rng(123)
    s = ts.random_spawn(2, rng)
    grid, heads, alive = state_to_arrays(s)
    p0_start = potential(grid, heads, alive, 0)
    assert 0.1 < p0_start < 0.9

    turns_seen = 0
    while not ts.is_terminal(s):
        p = s.next_to_move
        grid, heads, alive = state_to_arrays(s)
        for pid in range(s.num_players):
            phi = potential(grid, heads, alive, pid)
            assert np.isfinite(phi)
            assert 0.0 <= phi <= 1.0
        turns_seen += 1
        move = ts.heuristic_action(s, p)
        ts.step_player(s, p, move)
    assert turns_seen > 0
