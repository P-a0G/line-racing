import random
import numpy as np
from random import randint

height = 20
width = 30

directions = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}


class Environment:
    def __init__(self, player_1_pos, player_2_pos, player_3_pos):
        # todo
        self.height = 20
        self.width = 30

        self.grid = np.zeros((self.height, self.width))

        self.player_1_pos = player_1_pos
        self.player_2_pos = player_2_pos
        self.player_3_pos = player_3_pos

    def reset(self):
        """
            Reset world
        """
        self.grid = np.zeros((self.height, self.width))

        self.player_1_pos = randint(0, self.height - 1), randint(0, self.width - 1)
        self.player_2_pos = randint(0, self.height - 1), randint(0, self.width - 1)

        while self.player_2_pos == self.player_1_pos:
            self.player_2_pos = randint(0, self.height - 1), randint(0, self.width - 1)

        if random.random() > 0.5:
            self.player_3_pos = randint(0, self.height - 1), randint(0, self.width - 1)
            while self.player_3_pos == self.player_1_pos or self.player_3_pos == self.player_2_pos:
                self.player_3_pos = randint(0, self.height - 1), randint(0, self.width - 1)

        self.grid[self.player_3_pos[1]][self.player_3_pos[0]] = 3
        self.grid[self.player_2_pos[1]][self.player_2_pos[0]] = 2
        self.grid[self.player_1_pos[1]][self.player_1_pos[0]] = 1

    def step(self, action):
        """
            Action: 0, 1, 2, 3
        """
        # todo
        new_state = None
        recompense = 0
        return new_state, recompense

    def show(self):
        """
            Show the grid
        """
        # todo
        pass

    def is_finished(self):
        pass  # todo


def take_action(st, Q, eps):
    # Take an action
    if random.uniform(0, 1) < eps:
        action = randint(0, 3)
    else:  # Or greedy action
        action = np.argmax(Q[st])
    return action


