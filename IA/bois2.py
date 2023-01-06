import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.
import numpy as np


directions = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}

height = 20
width = 30
area = np.zeros((height, width), dtype=int)


def debug_print(*args):
    print(*args, file=sys.stderr, flush=True)


def print_field(field):
    for line in field:
        strg = ''
        for e in line:
            strg += str(e).ljust(3)
        debug_print(strg)
    debug_print("\n")


def is_in_map(x, y):
    return 0 <= x < height and 0 <= y < width


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

    print_field(area)
    debug_print("my pos:", my_position)

    x, y = my_position
    for e, (dy, dx) in directions.items():
        debug_print(is_in_map(y + dy, x + dx), y + dy, x + dx)
        if is_in_map(y + dy, x + dx) and area[y + dy, x + dx] == 0:
            print(e)
            break
