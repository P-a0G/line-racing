import random
import numpy as np
from random import randint
import time
from utils import load_bar

height = 20
width = 30

directions = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
actions = list(directions.values())


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
        reward = 0

        dx, dy = actions[action]
        x, y = self.player_pos[self.current_player]

        if self.is_move_possible(x + dx, y + dy):
            self.player_pos[self.current_player] = (x + dx, y + dy)
            self.grid[x + dx, y + dy] = self.current_player + 1  # 1, 2 or 3
        else:
            self.remove_player(self.current_player)
            reward -= 1

        # next player
        self.current_player = (self.current_player + 1) % 3
        if self.player_pos[self.current_player] is None:
            self.current_player = (self.current_player + 1) % 3

        new_state = get_state(self.grid, self.player_pos)

        return new_state, reward

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

    def is_move_possible(self, x, y):
        """check if the move is possible"""
        if x < 0 or y < 0 or x >= height or y >= width:
            return False

        return self.grid[x][y] == 0

    def remove_player(self, player_id):
        """player died, removing its walls and set its position to None"""
        # print("> Player", player_id + 1, 'lost !')
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
            if self.is_move_possible(x + dx, y + dy):
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
    if st in Q.keys():
        return np.argmax(Q[st])

    # don't have this state, select a random move
    possible_moves = env.get_moves()
    if len(possible_moves) == 0:
        return 0
    return random.choice(possible_moves)


def get_state(grid, player_pos):
    print("grid:\n", grid)
    print("player pos:", player_pos)

    return ""

# todo: https://github.com/thibo73800/aihub/blob/master/rl/sticks.py
class Player:
    def __init__(self, is_human, trainable=True):
        self.is_human = is_human
        self.history = []
        self.V = {}
        self.win_nb = 0
        self.lose_nb = 0
        self.rewards = []
        self.eps = 0.99
        self.trainable = trainable

    def reset_stat(self):
        # Reset stat
        self.win_nb = 0
        self.lose_nb = 0
        self.rewards = []

    def greedy_step(self, state):
        # greedy step

        # actions = [1, 2, 3]
        # vmin = None
        # vi = None
        # for i in range(0, 3):
        #     a = actions[i]
        #     if state - a > 0 and (vmin is None or vmin > self.V[state - a]):
        #         vmin = self.V[state - a]
        #         vi = i
        # return actions[vi if vi is not None else 1]

        return 0  # todo

    def play(self, state):
        # Play given the @state (int)
        if self.is_human is False:
            # Take random action
            if random.uniform(0, 1) < self.eps:
                action = randint(0, 3)
            else:  # Or greedy action
                action = self.greedy_step(state)
        else:
            action = int(input("$>"))
        return action

    def add_transition(self, n_tuple):
        # Add one transition to the history: tuple (s, a , r, s')
        self.history.append(n_tuple)
        s, a, r, sp = n_tuple
        self.rewards.append(r)

    def train(self):
        if not self.trainable or self.is_human is True:
            return

        # Update the value function if this player is not human
        for transition in reversed(self.history):
            s, a, r, sp = transition
            if r == 0:
                self.V[s] = self.V[s] + 0.001 * (self.V[sp] - self.V[s])
            else:
                self.V[s] = self.V[s] + 0.001 * (r - self.V[s])

        self.history = []


def play(game, p1, p2, p3=None, train=True):
    state = game.reset()
    players = [p1, p2]
    random.shuffle(players)
    p = 0
    while game.is_finished() is False:

        if players[p % 2].is_human:
            game.display()

        action = players[p % 2].play(state)
        n_state, reward = game.step(action)

        #  Game is over. Ass stat
        if (reward != 0):
            # Update stat of the current player
            players[p % 2].lose_nb += 1. if reward == -1 else 0
            players[p % 2].win_nb += 1. if reward == 1 else 0
            # Update stat of the other player
            players[(p + 1) % 2].lose_nb += 1. if reward == 1 else 0
            players[(p + 1) % 2].win_nb += 1. if reward == -1 else 0

        # Add the reversed reward and the new state to the other player
        if p != 0:
            s, a, r, sp = players[(p + 1) % 2].history[-1]
            players[(p + 1) % 2].history[-1] = (s, a, reward * -1, n_state)

        players[p % 2].add_transition((state, action, reward, None))

        state = n_state
        p += 1

    if train:
        p1.train()
        p2.train()


if __name__ == '__main__':
    learning_rate = 0.1
    env = Environment()
    st = env.reset()

    Q = {}

    n = 10_000
    start = time.time()
    for i in range(n):
        print(f'\rSimulating games: {load_bar(i, n)}', end="")
        state = env.reset()
        while not env.is_finished():
            # env.show()

            # action = take_action(state, Q, eps=0.4)
            action = take_action(state, Q, eps=1.0)

            next_state, recompense = env.step(action)
            # print("next state:", next_state)
            # print("recompense:", recompense)

            # update Q function
            next_action = take_action(next_state, Q, 0.0)
            if state not in Q.keys():
                Q[state] = [0, 0, 0, 0]
            Q[state][action] += learning_rate * (recompense + (1 - learning_rate) * Q[next_state][next_action] - Q[state][action])

            state = next_state
    print("\n")
    print("Total time:", time.time() - start, "s")

    # print("Q table:")
    # print(Q)

    """
    game = StickGame(12)

    # PLayers to train
    p1 = StickPlayer(is_human=False, size=12, trainable=True)
    p2 = StickPlayer(is_human=False, size=12, trainable=True)
    # Human player and random player
    human = StickPlayer(is_human=True, size=12, trainable=False)
    random_player = StickPlayer(is_human=False, size=12, trainable=False)

    # Train the agent
    for i in range(0, 10000):
        if i % 10 == 0:
            p1.eps = max(p1.eps*0.996, 0.05)
            p2.eps = max(p2.eps*0.996, 0.05)
        play(game, p1, p2)
    p1.reset_stat()

    # Display the value function
    for key in p1.V:
        print(key, p1.V[key])
    print("--------------------------")

    # Play agains a random player
    for _ in range(0, 1000):
        play(game, p1, random_player, train=False)
    print("p1 win rate", p1.win_nb/(p1.win_nb + p1.lose_nb))
    print("p1 win mean", np.mean(p1.rewards))

    # Play agains us
    while True:
        play(game, p1, human, train=False)
    """