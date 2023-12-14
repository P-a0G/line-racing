import os
import random
import numpy as np
import time
from utils import load_bar
from matplotlib import pyplot as plt
from random import randint

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
            reward = -1

        # next player
        self.current_player = (self.current_player + 1) % 3
        if self.player_pos[self.current_player] is None:
            self.current_player = (self.current_player + 1) % 3

        return reward

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

    def is_dead(self, player_idx):
        return self.player_pos[player_idx] is None


class Player:
    def __init__(self, is_human, idx=-1, trainable=True, state_dict_pth=""):
        self.is_human = is_human
        self.idx = idx
        self.history = []
        self.V = {}
        self.win_nb = 0
        self.lose_nb = 0
        self.rewards = []
        self.eps = 0.99
        self.trainable = trainable

        self.load_state_dict(state_dict_pth)

    def reset_stat(self):
        # Reset stat
        self.win_nb = 0
        self.lose_nb = 0
        self.rewards = []

    def greedy_step(self, grid, player_pos, my_idx):  # todo
        # greedy step

        state = self.get_state(grid, player_pos, my_idx)

        acts = self.get_moves(grid, player_pos[my_idx])

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

    def play(self, grid, player_pos, my_idx):
        if self.is_human:
            # human action
            action = int(input("$>"))
            print("chosen action:", list(directions.keys())[action])
            return action

        assert player_pos is not None, "Error player is dead"

        if random.uniform(0, 1) < self.eps:
            # random action
            possible_moves = self.get_moves(grid, player_pos[my_idx])

            return random.choice(possible_moves)

        # greedy action
        return self.greedy_step(grid, player_pos, my_idx)

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
                self.V[s] = self.V.get(s, 0.0) + 0.001 * (self.V.get(sp, 0.0) - self.V.get(s, 0.0))
            else:
                self.V[s] = self.V.get(s, 0.0) + 0.001 * (r - self.V.get(s, 0.0))

        self.history = []

    def get_moves(self, grid, player_pos):
        possible_moves = []

        if player_pos is None:
            assert False, f"Error idx={self.idx} player_pos={player_pos}"
        x, y = player_pos
        for k in range(4):
            dx, dy = actions[k]
            if height > x + dx >= 0 and width > y + dy >= 0 and grid[x + dx][y + dy] == 0:
                possible_moves.append(k)

        if len(possible_moves) == 0:
            return [0]
        return possible_moves

    @staticmethod
    def get_state(grid, player_pos, my_idx):
        encoded_map = []

        # load player positions
        for a in range(3):
            if player_pos[(my_idx + a) % 3] is None:
                p_x, p_y = -1, -1
            else:
                p_x, p_y = player_pos[(my_idx + a) % 3]

            encoded_map.append(p_x)
            encoded_map.append(p_y)

        # load the map
        for u in range(width):
            encoded_line_p1 = 0
            encoded_line_p2 = 0
            encoded_line_p3 = 0
            for v in range(height):
                if grid[v][u] == 1:
                    encoded_line_p1 += 1 << u
                if grid[v][u] == 2:
                    encoded_line_p2 += 1 << u
                if grid[v][u] == 3:
                    encoded_line_p3 += 1 << u

            encoded_map.append(encoded_line_p1)
            encoded_map.append(encoded_line_p2)
            encoded_map.append(encoded_line_p3)

        encoded_map = tuple(encoded_map)

        return encoded_map

    def save_state_dict(self, pth):
        with open(pth, "w") as f:
            for k, v in self.V.items():
                f.write(f"{k};{v}\n")

    def load_state_dict(self, pth):
        if os.path.exists(pth):
            with open(pth, "r") as f:
                lines = f.readlines()

            for line in lines:
                k, v = line.split(";")
                k = tuple(int(num) for num in k.replace('(', '').replace(')', '').replace('...', '').split(', '))
                self.V[k] = float(v)


def play(game, p1, p2, p3=None, train=True):
    game.reset()
    players = [p1, p2]
    random.shuffle(players)  # random player order
    p = 0
    nb_players = len(players)
    while not game.is_finished():
        if game.player_pos[p % nb_players] is None:  # player is dead
            p += 1

        if players[p % nb_players].is_human:
            game.show()

        action = players[p % nb_players].play(game.grid, game.player_pos, game.current_player)
        player_state = players[p % nb_players].get_state(game.grid, game.player_pos, game.current_player)
        reward = game.step(action)

        #  A player lost
        if reward != 0:
            # todo vérifier si le code va dans le while
            # Add the reversed reward to the winner
            if not game.is_dead((p + 1) % nb_players) and game.is_dead((p + 2) % nb_players):  # p + 2 wins
                if len(players[(p + 1) % nb_players].history) == 0:
                    game.show()
                    assert False
                s, a, r, sp = players[(p + 1) % nb_players].history[-1]
                n_state = players[(p + 1) % nb_players].get_state(game.grid, game.player_pos, game.current_player)
                players[(p + 1) % nb_players].history[-1] = (s, a, reward * -1, n_state)
            else:  # p + 1 wins
                if len(players[(p + 2) % nb_players].history) == 0:
                    game.show()
                    assert False
                s, a, r, sp = players[(p + 2) % nb_players].history[-1]
                n_state = players[(p + 2) % nb_players].get_state(game.grid, game.player_pos, game.current_player)
                players[(p + 2) % nb_players].history[-1] = (s, a, reward * -1, n_state)

        players[p % nb_players].add_transition((player_state, action, reward, None))

        p += 1

    # Update stat of players
    for i in range(nb_players):
        if game.player_pos[i] is None:
            players[i].lose_nb += 1
        else:
            players[i].win_nb += 1

    if train:
        for player in players:
            player.train()

    return p  # nb of turns


if __name__ == '__main__':
    game = Game()

    p1_save = "player1_save.csv"
    p2_save = "player2_save.csv"

    # PLayers to train
    p1 = Player(is_human=False, idx=0, trainable=True, state_dict_pth=p1_save)
    p2 = Player(is_human=False, idx=1, trainable=True, state_dict_pth=p2_save)

    # Random player
    random_player = Player(is_human=False, idx=2, trainable=False)

    # Train the agent
    n = 100_000
    start = time.time()
    game_turns = []
    avr_turn_list = []
    for i in range(n):
        if len(game_turns) > 0:
            avr_turns = int(sum(game_turns[-100:]) / len(game_turns[-100:]))
            avr_turn_list.append(avr_turns)
        else:
            avr_turns = None
        print(f'\rSimulating games: {load_bar(i, n, start_time=start)}, avr turn = {avr_turns}', end="")
        if i % 10 == 0:
            p1.eps = max(p1.eps*0.996, 0.05)
            p2.eps = max(p2.eps*0.996, 0.05)
        nb_turns = play(game, p1, p2)
        game_turns.append(nb_turns)

        if i % 10_000 == 0:
            p1.save_state_dict(p1_save)
            p2.save_state_dict(p2_save)

    print("\n")

    p1.reset_stat()
    print("\n")
    print("Simulation time:", round(time.time() - start, 2), "s")

    plt.plot(game_turns)
    plt.show()

    plt.plot(avr_turn_list)
    plt.show()

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

