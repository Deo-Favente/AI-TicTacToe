from random import randint
import random
import numpy as np

"""
Simple implementation for the tic tac toe game AI using the value function.
The initial project was made by Thibault Neuveu
https://github.com/thibo73800/aihub/blob/master/rl/sticks.py
"""


class  TicTacToe(object):
    """
        Tic Tac Toe Game class.
    """

    def __init__(self, size):
        # @size size of the grid
        super(TicTacToe, self).__init__()
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.cells_played = []
        self.size = size

    def is_finished(self):
        # Check if the game is over @return Boolean
        for i in range(0, self.size):
            for j in range(0, self.size):
                if self.grid[i][j] == 0:
                    return False
        return True

    def reset(self):
        # Reset the state of the game
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.cells_played = []
        return self.grid

    def display(self):
        # Display the state of the game
        for l in self.grid:
            print("|", end=" ")
            for v in l:
                print(v, "|", end=" ")
            print("")
        print("")

    def step(self, action, player):
        # @action int between 0 and 8, the action to play.
        # the action must be in an empty cell
        # @return the new state and the reward
        i = action // self.size
        j = action % self.size
        self.grid[i][j] = player
        self.cells_played.append(action)
        if self.is_finished():
            # check who won
            pass
        return self.grid, 0
        

class StickPlayer(object):
    """
        Stick Player
    """

    def __init__(self, is_human, size, trainable=True):
        # @nb Number of stick to play with
        super(StickPlayer, self).__init__()
        self.is_human = is_human
        self.history = []
        self.V = {}
        for s in range(1, size+1):
            self.V[s] = 0.
        self.win_nb = 0.
        self.lose_nb = 0.
        self.rewards = []
        self.eps = 0.99
        self.trainable = trainable

    def reset_stat(self):
        # Reset stat
        self.win_nb = 0
        self.lose_nb = 0
        self.rewards = []

    def greedy_step(self, state):
        # Greedy step
        actions = [1, 2, 3]
        vmin = None
        vi = None
        for i in range(0, 3):
            a = actions[i]
            if state - a > 0 and (vmin is None or vmin > self.V[state - a]):
                vmin = self.V[state - a]
                vi = i
        return actions[vi if vi is not None else 1]

    def play(self, state):
        # PLay given the @state (int)
        if self.is_human is False:
            # Take random action
            if random.uniform(0, 1) < self.eps:
                action = randint(1, 3)
            else: # Or greedy action
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
                self.V[s] = self.V[s] + 0.001*(self.V[sp] - self.V[s])
            else:
                self.V[s] = self.V[s] + 0.001*(r - self.V[s])

        self.history = []

def play(game, p1, p2, train=True):
    state = game.reset()
    players = [p1, p2]
    random.shuffle(players)
    p = 0
    while game.is_finished() is False:

        if players[p%2].is_human:
            game.display()

        action = players[p%2].play(state)
        n_state, reward = game.step(action)

        #  Game is over. Ass stat
        if (reward != 0):
            # Update stat of the current player
            players[p%2].lose_nb += 1. if reward == -1 else 0
            players[p%2].win_nb += 1. if reward == 1 else 0
            # Update stat of the other player
            players[(p+1)%2].lose_nb += 1. if reward == 1 else 0
            players[(p+1)%2].win_nb += 1. if reward == -1 else 0

        # Add the reversed reward and the new state to the other player
        if p != 0:
            s, a, r, sp = players[(p+1)%2].history[-1]
            players[(p+1)%2].history[-1] = (s, a, reward * -1, n_state)

        players[p%2].add_transition((state, action, reward, None))

        state = n_state
        p += 1

    if train:
        p1.train()
        p2.train()

if __name__ == '__main__':
    game = TicTacToe(3)
    game.display()
    game.step(1, "X")
    game.step(3, "O")
    game.display()

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
