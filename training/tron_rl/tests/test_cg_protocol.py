import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "submission"))
from cg_main import parse_round, legal_moves  # noqa: E402


def test_trail_reconstruction_matches_rules_md_worked_example():
    # The exact 4-turn transcript from rules.md's own worked example: player
    # 0 goes RIGHT then UP, player 1 goes UP (so far) -- verifies the
    # incremental (X0,Y0)/(X1,Y1)-only protocol reconstructs the full
    # polyline losslessly, not just the latest head position.
    trails = {}

    grid, heads, alive = parse_round(2, ["9 5 9 5", "10 7 10 7"], trails)
    assert trails == {0: {(9, 5)}, 1: {(10, 7)}}
    assert heads == [(9, 5), (10, 7)]
    assert alive == [True, True]

    grid, heads, alive = parse_round(2, ["9 5 10 5", "10 7 10 7"], trails)
    assert trails == {0: {(9, 5), (10, 5)}, 1: {(10, 7)}}
    assert heads == [(10, 5), (10, 7)]

    grid, heads, alive = parse_round(2, ["9 5 10 5", "10 7 10 6"], trails)
    assert trails == {0: {(9, 5), (10, 5)}, 1: {(10, 7), (10, 6)}}
    assert heads == [(10, 5), (10, 6)]

    grid, heads, alive = parse_round(2, ["9 5 10 4", "10 7 10 6"], trails)
    assert trails == {0: {(9, 5), (10, 5), (10, 4)}, 1: {(10, 7), (10, 6)}}
    assert heads == [(10, 4), (10, 6)]
    assert grid[5, 9] == 0 and grid[5, 10] == 0 and grid[4, 10] == 0  # player 0's full path
    assert grid[7, 10] == 1 and grid[6, 10] == 1                       # player 1's path so far


def test_elimination_clears_the_players_trail():
    trails = {0: {(5, 5), (6, 5)}, 1: {(10, 10)}}
    grid, heads, alive = parse_round(2, ["5 5 6 5", "-1 -1 -1 -1"], trails)
    assert 1 not in trails
    assert alive == [True, False]
    assert heads[1] is None
    assert grid[5, 6] == -1 or grid[5, 5] == 0  # player 1's old cells are free again, not claimed by anyone
    assert grid[10, 10] == -1


def test_three_player_round_with_one_already_eliminated():
    trails = {0: {(1, 1)}, 2: {(20, 15)}}
    grid, heads, alive = parse_round(3, ["1 1 1 1", "-1 -1 -1 -1", "20 15 20 15"], trails)
    assert alive == [True, False, True]
    assert heads[1] is None
    assert 1 not in trails
    assert grid[1, 1] == 0
    assert grid[15, 20] == 2


def test_legal_moves_respects_bounds_and_occupancy():
    grid = np.full((20, 30), -1, dtype=np.int8)
    grid[0, 5] = 0  # occupied cell directly above (5,1)... i.e. (x=5,y=0)
    legal = legal_moves(grid, 5, 1)  # UP, DOWN, LEFT, RIGHT
    assert legal[0] == False  # UP into the occupied cell (5,0)
    assert legal[1] == True
    assert legal[2] == True
    assert legal[3] == True

    corner_legal = legal_moves(grid, 0, 0)
    assert corner_legal[0] == False  # UP off the top edge
    assert corner_legal[2] == False  # LEFT off the left edge
