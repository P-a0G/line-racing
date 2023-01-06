import sys
import math
import random
import numpy as np
import time

directions = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}

height = 20
width = 30
area = np.zeros((height, width), dtype=int)


def debug_print(*args):
    print(*args, file=sys.stderr, flush=True)


def print_field(field):
    for line in field:
        strg = ''
        for e in line:
            strg += str(e).ljust(1) if e > 0 else '.'.ljust(1)
        debug_print(strg)
    debug_print("\n")


def is_in_map(x, y):
    return 0 <= x < height and 0 <= y < width


def node_score(w, n, N):
    if n == 0:
        return 1
    return w/n + 1.4142135 * (math.log2(N) / n) ** 0.5


class Tree:
    def __init__(self, game_state, player_turn):
        self.win = 0
        self.n = 0
        self.children = []
        self.game_state = game_state
        self.player_turn = player_turn  # 0 -> n

    def is_leaf(self):
        return len(self.children) == 0

    def get_score(self, N):
        return node_score(self.win, self.n, N)

    def init_children(self):
        self.children = []  # todo

    def selection(self):
        if self.is_leaf():
            self.init_children()
            if len(self.children) == 0:
                pass  # todo  # perdu, pas de coup possible

    def get_best_move(self):
        n_simu = 0
        while time.time() - start < 0.099:
            # selection
            self.selection()  # todo

            # expansion

            # simulation

            # backpropagation
            while time.time() - start < 0.099:
                winner, turn = simulate_game(
                    game_state=self.game_state.copy(),
                    player_turn=p,
                    player_0_pos=players_pos[0],
                    player_1_pos=players_pos[1],
                    player_2_pos=players_pos[2],
                    player_3_pos=players_pos[3]
                )
                n_simu += 1

            debug_print("n_simu=", n_simu)
            debug_print("winner board:", winner_board)
            debug_print("avr turn:", int(total_turn / n_simu))
            # print_field(area)
            debug_print("my pos:", my_position)

        return 'UP'


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


# game loop
while True:
    # n: total number of players (2 to 4).
    # p: your player number (0 to 3).
    n, p = [int(i) for i in input().split()]

    my_position = None
    players_pos = [None] * 4

    for i in range(n):
        # x0: starting X coordinate of lightcycle (or -1)
        # y0: starting Y coordinate of lightcycle (or -1)
        # x1: starting X coordinate of lightcycle (can be the same as X0 if you play before this player)
        # y1: starting Y coordinate of lightcycle (can be the same as Y0 if you play before this player)
        x0, y0, x1, y1 = [int(j) for j in input().split()]
        players_pos[i] = (x1, y1)
        start = time.time()
        area[y0, x0] = i + 1
        area[y1, x1] = i + 1

        if i == p:
            my_position = x1, y1

    tree = Tree(area.copy(), p)

    move = tree.get_best_move()

    print(move)
