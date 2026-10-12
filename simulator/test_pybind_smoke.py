import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import tronsim as ts


def test_random_spawn_distinct_cells():
    rng = ts.Rng(42)
    s = ts.random_spawn(3, rng)
    heads = list(s.head_cell)
    assert len(set(heads)) == 3
    assert all(s.cell_owner[h] == p for p, h in enumerate(heads))


def test_full_playthrough_reaches_exactly_one_winner():
    rng = ts.Rng(1)
    s = ts.random_spawn(2, rng)
    moves = 0
    while not ts.is_terminal(s):
        p = s.next_to_move
        legal = ts.legal_moves(s, p)
        directions = [ts.UP, ts.DOWN, ts.LEFT, ts.RIGHT]
        chosen = next((d for d, ok in zip(directions, legal) if ok), ts.UP)
        ts.step_player(s, p, chosen)
        moves += 1
        assert moves < 2000  # safety bound for this smoke test, not the engine's own MAX_MOVES
    assert ts.alive_count(s) == 1
    assert ts.winner(s) in (0, 1)


def test_heuristic_opponent_outlasts_random_moves():
    rng = ts.Rng(7)
    wins = {0: 0, 1: 0}
    for seed in range(20):
        g = ts.Rng(seed)
        s = ts.random_spawn(2, g)
        while not ts.is_terminal(s):
            p = s.next_to_move
            if p == 0:
                move = ts.heuristic_action(s, p)
            else:
                legal = ts.legal_moves(s, p)
                directions = [ts.UP, ts.DOWN, ts.LEFT, ts.RIGHT]
                move = next((d for d, ok in zip(directions, legal) if ok), ts.UP)
            ts.step_player(s, p, move)
        if ts.winner(s) != -1:
            wins[ts.winner(s)] += 1
    assert wins[0] > wins[1]  # the space-maximizing heuristic should beat first-legal-move


if __name__ == "__main__":
    test_random_spawn_distinct_cells()
    test_full_playthrough_reaches_exactly_one_winner()
    test_heuristic_opponent_outlasts_random_moves()
    print("All pybind smoke tests passed.")
