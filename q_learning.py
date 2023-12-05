import random
import numpy as np
from random import randint
import time

height = 20
width = 30

directions = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
actions = list(directions.values())


def load_bar(i, n):
    end = f"{i + 1}/{n}"
    i = (i + 1) * 100 // n
    bar = f"[{'■' * i}{' ' * (100 - i)}]\t {end}"
    return bar


class Environment:
    def __init__(self):
        self.height = 20
        self.width = 30

        self.grid = np.zeros((self.height, self.width))

        self.current_player = 0

        self.player_pos = [
            (randint(0, self.height - 1), randint(0, self.width - 1)),
            (randint(0, self.height - 1), randint(0, self.width - 1)),
            (randint(0, self.height - 1), randint(0, self.width - 1)) if random.random() > 0.5 else None
        ]

        while self.player_pos[1] == self.player_pos[0]:
            self.player_pos[1] = (randint(0, self.height - 1), randint(0, self.width - 1))

        if self.player_pos[2] is not None:
            while self.player_pos[2] == self.player_pos[0] or self.player_pos[2] == self.player_pos[1]:
                self.player_pos[2] = (randint(0, self.height - 1), randint(0, self.width - 1))

        if self.player_pos[2] is not None:
            self.grid[self.player_pos[2][0]][self.player_pos[2][1]] = 3

        self.grid[self.player_pos[1][0]][self.player_pos[1][1]] = 2
        self.grid[self.player_pos[0][0]][self.player_pos[0][1]] = 1

    def reset(self):
        """
            Reset world
        """
        self.grid = np.zeros((self.height, self.width))

        self.current_player = 0

        self.player_pos[0] = (randint(0, self.height - 1), randint(0, self.width - 1))
        self.player_pos[1] = (randint(0, self.height - 1), randint(0, self.width - 1))

        while self.player_pos[1] == self.player_pos[0]:
            self.player_pos[1] = (randint(0, self.height - 1), randint(0, self.width - 1))

        if random.random() > 0.5:
            self.player_pos[2] = (randint(0, self.height - 1), randint(0, self.width - 1))
            while self.player_pos[2] == self.player_pos[0] or self.player_pos[2] == self.player_pos[1]:
                self.player_pos[2] = (randint(0, self.height - 1), randint(0, self.width - 1))

        if self.player_pos[2] is not None:
            self.grid[self.player_pos[2][0]][self.player_pos[2][1]] = 3
        self.grid[self.player_pos[1][0]][self.player_pos[1][1]] = 2
        self.grid[self.player_pos[0][0]][self.player_pos[0][1]] = 1

        state = 0  # todo def and return state
        return state

    def step(self, action):
        """
            Action: 0, 1, 2, 3
        """
        new_state = None
        recompense = 0

        dx, dy = actions[action]
        x, y = self.player_pos[self.current_player]

        if self.is_good_move(x + dx, y + dy):
            self.player_pos[self.current_player] = (x + dx, y + dy)
            self.grid[x + dx, y + dy] = self.current_player + 1  # 1, 2 or 3
        else:
            self.remove_player(self.current_player)

        # next player
        self.current_player = (self.current_player + 1) % 3
        if self.player_pos[self.current_player] is None:
            self.current_player = (self.current_player + 1) % 3

        return new_state, recompense

    def show(self):
        """
            Show the grid
        """
        for k in range(height):
            line_str = ''
            for l in range(width):
                if self.grid[k][l] == 0:
                    line_str += " . "
                elif (k, l) == self.player_pos[0]:
                    line_str += ' ➀ '
                elif (k, l) == self.player_pos[1]:
                    line_str += ' ② '
                elif (k, l) == self.player_pos[2]:
                    line_str += ' ③ '
                else:
                    line_str += " " + str(int(self.grid[k][l])) + " "
            print(line_str)
        print("\n")

    def is_finished(self):
        return (self.player_pos[0] is None and self.player_pos[1] is None)\
               or (self.player_pos[0] is None and self.player_pos[2] is None)\
               or (self.player_pos[2] is None and self.player_pos[1] is None)

    def is_good_move(self, x, y):
        """check if the move is possible"""
        if x < 0 or y < 0 or x >= height or y >= width:
            return False

        return self.grid[x][y] == 0

    def remove_player(self, player_id):
        """player died, removing its walls and set its position to None"""
        print("> Player", player_id + 1, 'lost !')
        self.player_pos[player_id] = None

        for i in range(height):
            for j in range(width):
                if self.grid[i][j] == player_id + 1:
                    self.grid[i][j] = 0

    def get_moves(self):
        possible_moves = []

        x, y = self.player_pos[self.current_player]
        for k in range(4):
            dx, dy = actions[k]
            if self.is_good_move(x + dx, y + dy):
                possible_moves.append(k)

        return possible_moves


def take_action(st, Q, eps):
    # Take an action
    if random.uniform(0, 1) < eps:
        possible_moves = env.get_moves()
        if len(possible_moves) == 0:
            return 0
        return random.choice(possible_moves)
    # Or greedy action
    return np.argmax(Q[st])  # todo


if __name__ == '__main__':
    learning_rate = 0.1
    env = Environment()
    st = env.reset()

    Q = {}

    n = 10_000
    start = time.time()
    for i in range(n):
        print(f'\rSimulationg games: {load_bar(i, n)}')
        state = env.reset()
        while not env.is_finished():
            # env.show()

            # action = take_action(state, Q, eps=0.4)
            action = take_action(state, Q, eps=1.0)

            next_state, recompense = env.step(action)
            # print("next state:", next_state)
            # print("recompense:", recompense)

            # update Q function
            # next_action = take_action(next_state, Q, 0.0)
            # Q[state][action] += learning_rate * (recompense + (1 - learning_rate) * Q[next_state][next_action] - Q[state][action])

            state = next_state

    print("Total time:", time.time() - start, "s")

    # print("Q table:")
    # print(Q)

