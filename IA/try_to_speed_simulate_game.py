import random
import time
from numba import njit
import numpy as np


directions = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}


def is_in_map(x, y):
    return 0 <= x < 20 and 0 <= y < 30


# @njit
def simulate_game(game_state, player_turn, player_0_pos, player_1_pos, player_2_pos, player_3_pos):
    players = [0, 0, 0, 0]
    players_pos = [player_0_pos, player_1_pos, player_2_pos, player_3_pos]
    if player_0_pos is not None:
        players[0] = 1
    if player_1_pos is not None:
        players[1] = 1
    if player_2_pos is not None:
        players[2] = 1
    if player_3_pos is not None:
        players[3] = 1

    player_id = player_turn
    turn = 0
    while True:
        turn += 1
        if turn > 600:
            assert False  # turn > 600
        i, j = players_pos[player_id]
        moves = [
            (i + dx, j + dy) for (dy, dx) in directions.values()
            if is_in_map(j + dy, i + dx) and game_state[j + dy, i + dx] == 0
        ]

        # player lost
        if len(moves) == 0:
            players[player_id] = 0
            if sum(players) == 1:
                break
            game_state[game_state == player_id + 1] = 0
            assert sum(players) > 1

        # select a random moves
        else:
            new_i, new_j = random.choice(moves)
            players_pos[player_id] = (new_i, new_j)
            game_state[new_j, new_i] = player_id + 1

        # get next player
        while True:
            player_id = (player_id + 1) % 4
            if players[player_id]:
                break

    assert sum(players) == 1
    return players.index(1), turn


if __name__ == '__main__':
    height = 20
    width = 30
    area = np.zeros((height, width), dtype=int)

    x1 = random.randint(0, width - 1)
    y1 = random.randint(0, height - 1)

    x2 = random.randint(0, width - 1)
    y2 = random.randint(0, height - 1)

    while x1 == x2 and y1 == y2:
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)

    area[y1, x1] = 1
    area[y2, x2] = 2

    players_pos = [(x1, y1), (x2, y2), None, None]

    start = time.time()

    n_simu = 0
    winner_board = [0, 0, 0, 0]
    total_turn = 0
    while time.time() - start < 0.0995:
        winner, turn = simulate_game(
            game_state=area.copy(),
            player_turn=1,
            player_0_pos=players_pos[0],
            player_1_pos=players_pos[1],
            player_2_pos=players_pos[2],
            player_3_pos=players_pos[3]
        )
        n_simu += 1
        winner_board[winner] += 1
        total_turn += turn

    print("time:", time.time() - start)

    print("n_simu=", n_simu)
    print("winner board:", winner_board)
    print("avr turn:", total_turn / n_simu)