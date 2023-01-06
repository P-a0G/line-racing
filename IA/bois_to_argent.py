import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
import numpy as np


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
            strg += str(e).ljust(3) if e > 0 else '.'.ljust(3)
        debug_print(strg)
    debug_print("\n")


def is_in_map(x, y):
    return 0 <= x < height and 0 <= y < width


def n_accessible(map, start_pos):
    n = 0
    pile = [start_pos]
    while len(pile) > 0:
        x, y = pile.pop()
        for dx, dy in directions.values():
            new_x, new_y = x + dx, y + dy
            if is_in_map(new_y, new_x) and map[new_y, new_x] == 0:
                pile.append((new_x, new_y))
                map[new_y, new_x] = 1
                n += 1
    return n


# game loop
while True:
    # n: total number of players (2 to 4).
    # p: your player number (0 to 3).
    n, p = [int(i) for i in input().split()]

    my_position = None

    for i in range(n):
        # x0: starting X coordinate of lightcycle (or -1)
        # y0: starting Y coordinate of lightcycle (or -1)
        # x1: starting X coordinate of lightcycle (can be the same as X0 if you play before this player)
        # y1: starting Y coordinate of lightcycle (can be the same as Y0 if you play before this player)
        x0, y0, x1, y1 = [int(j) for j in input().split()]
        area[y0, x0] = 1
        area[y1, x1] = 1

        if i == p:
            my_position = x1, y1

    # print_field(area)
    debug_print("my pos:", my_position)

    x, y = my_position
    best_dir = None
    max_n = -1

    for e, (dx, dy) in directions.items():
        new_x, new_y = x + dx, y + dy
        debug_print("check pos", e, new_x, new_y)
        if is_in_map(new_y, new_x) and area[new_y, new_x] == 0:
            n = n_accessible(area.copy(), (new_x, new_y))
            debug_print("\t n=", n)
            if n > max_n:
                best_dir = e
                max_n = n

    if max_n > -1:
        debug_print("Check OK")

    print(best_dir if best_dir is not None else 'UP')
