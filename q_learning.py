import random
import numpy as np
from random import randint
import time
from utils import load_bar

height = 20
width = 30

directions = {'UP': (-1, 0), 'RIGHT': (0, 1), 'DOWN': (1, 0), 'LEFT': (0, -1)}
actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class Game:
    def __init__(self):
        self.height = 20
        self.width = 30

        self.grid = np.zeros((self.height, self.width))

        self.current_player = 0

        self.player_pos = [
            (randint(0, self.height - 1), randint(0, self.width - 1)),
            (randint(0, self.height - 1), randint(0, self.width - 1)),
            None  # (randint(0, self.height - 1), randint(0, self.width - 1)) if random.random() > 0.5 else None
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

        self.player_pos = [
            (randint(0, self.height - 1), randint(0, self.width - 1)),
            (randint(0, self.height - 1), randint(0, self.width - 1)),
            None  # (randint(0, self.height - 1), randint(0, self.width - 1)) if random.random() > 0.5 else None
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

        new_state = self.get_state()

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

    def get_state(self):
        # todo
        # self.grid
        # self.player_pos

        return ""

    def is_dead(self, player_idx):
        return self.player_pos[player_idx] is None


class Player:
    def __init__(self, is_human, idx=-1, trainable=True):
        self.is_human = is_human
        self.idx = idx
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

    def greedy_step(self, grid, player_pos):
        # greedy step

        state = self.get_state(grid, player_pos)

        acts = self.get_moves(grid, player_pos)

        action = random.choice(acts)
        # vmin = None
        # vi = None
        # for i in range(0, 3):
        #     a = actions[i]
        #     if state - a > 0 and (vmin is None or vmin > self.V[state - a]):
        #         vmin = self.V[state - a]
        #         vi = i
        # return actions[vi if vi is not None else 1]

        return action

    def play(self, grid, player_pos):
        if self.is_human:
            # human action
            action = int(input("$>"))
            print("chosen action:", list(directions.keys())[action])
            return action

        assert player_pos is not None, "Error player is dead"

        if random.uniform(0, 1) < self.eps:
            # random action
            possible_moves = self.get_moves(grid, player_pos)
            if len(possible_moves) > 0:
                return random.choice(possible_moves)
            return 0

        # greedy action
        return self.greedy_step(grid, player_pos)

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

    def get_moves(self, grid, player_pos):
        possible_moves = []

        if player_pos[self.idx] is None:
            assert False, f"Error idx={self.idx} player_pos={player_pos}"
        x, y = player_pos[self.idx]
        for k in range(4):
            dx, dy = actions[k]
            if height > x + dx > 0 and width > y + dy > 0 and grid[x + dx][y + dy] == 0:
                possible_moves.append(k)

        return possible_moves

    def get_state(self, grid, player_pos):
        # todo
        # self.grid
        # self.player_pos

        return ""


def play(game, p1, p2, p3=None, train=True):
    state = game.reset()
    players = [p1, p2]
    random.shuffle(players)
    p = randint(0, len(players) - 1)  # random first player
    nb_players = len(players)
    while not game.is_finished():
        if game.player_pos[p % nb_players] is None:  # player is dead
            p += 1

        if players[p % nb_players].is_human:
            game.show()

        action = players[p % nb_players].play(game.grid, game.player_pos)
        n_state, reward = game.step(action)

        # #  A player lost
        # if reward != 0:
        # # todo vérifier si le code va dans le while
        # # Add the reversed reward to the winner
        # if not game.is_dead((p + 1) % nb_players) and game.is_dead((p + 2) % nb_players):  # p + 2 wins
        #     s, a, r, sp = players[(p + 1) % nb_players].history[-1]
        #     players[(p + 1) % nb_players].history[-1] = (s, a, reward * -1, n_state)
        # else:  # p + 1 wins
        #     s, a, r, sp = players[(p + 2) % nb_players].history[-1]
        #     players[(p + 2) % nb_players].history[-1] = (s, a, reward * -1, n_state)
        #
        # players[p % nb_players].add_transition((state, action, reward, None))

        state = n_state
        p += 1

    # Update stat of players
    for i in range(nb_players):
        if game.player_pos[i] is None:
            players[i].lose_nb += 1
        else:
            players[i].win_nb += 1

    if train:
        for p in players:
            p.train()


if __name__ == '__main__':
    game = Game()

    # PLayers to train
    p1 = Player(is_human=False, idx=0, trainable=True)
    p2 = Player(is_human=False, idx=1, trainable=True)

    # Random player
    random_player = Player(is_human=False, idx=2, trainable=False)

    # Train the agent
    n = 1
    start = time.time()
    for i in range(n):
        print(f'\rSimulating games: {load_bar(i, n)}', end="")
        if i % 10 == 0:
            p1.eps = max(p1.eps*0.996, 0.05)
            p2.eps = max(p2.eps*0.996, 0.05)
        play(game, p1, p2)
    print("\n")

    p1.reset_stat()
    print("\n")
    print("Simulation time:", round(time.time() - start, 2), "s")

    # # Display the value function
    # for key in p1.V:
    #     print(key, p1.V[key])

    # print("--------------------------")
    #
    # # Play against a random player
    # n = 100
    # start = time.time()
    # for i in range(n):
    #     print(f'\rPlaying against random player: {load_bar(i, n)}', end="")
    #     play(game, p1, random_player, train=False)
    # print("p1 win rate", p1.win_nb/(p1.win_nb + p1.lose_nb))
    # print("p1 win mean", np.mean(p1.rewards))
    # print("\n")
    # print("Simulation time:", time.time() - start, "s")

    # # Play agains us
    # # human player
    # human = Player(is_human=True, trainable=False)
    # p1.eps = 0  # no more greedy steps
    # play(game, p1, human, train=False)

