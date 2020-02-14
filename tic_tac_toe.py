# Python 3.8
import numpy as np
import collections
import time
import pickle
import copy
from random import *
import random

P1_SYMBOL = "1"
P2_SYMBOL = "2"
ROWS = 3
EMPTY_CELL = 0
COLUMNS = 3
ME_0_TO_WIN = 0
ME_1_TO_WIN = 1
ME_2_TO_WIN = 2
OP_0_TO_WIN = 3
OP_1_TO_WIN = 4
OP_2_TO_WIN = 5
class TicTacToe:

    def __init__(self, p1, p2):
        self.board = np.zeros((ROWS,COLUMNS))
        self.p1 = p1
        self.p2 = p2
        self.finished = False
        self.turn = p1
        self.winner = None
        self.turns = 0

    def has_ended(self):
        if self.who_won() is not None:
            self.winner = self.who_won()
            return True
        elif self.turns == 9:
            return True
        else:
            return False

    def reward(self, winner, loser):
        winner.reward(1)
        loser.reward(0)

    def who_won(self):
        if self.exists_winning_combination():
            return self.last_player_to_play()
        else:
            return None

    def last_player_to_play(self):
        if (self.turns % 2 == 0):
            return self.p2
        else:
            return self.p1

    def avaliable_positions(self):
        positions = []
        for i in range(ROWS):
            for j in range(COLUMNS):
                if self.board[i,j] == 0:
                    positions.append((i,j))
        return positions

    def exists_winning_combination(self):
        lines = self.lines()
        for line in lines:
            if np.array_equal(line, np.full(COLUMNS, float(P1_SYMBOL))) or np.array_equal(line, np.full(COLUMNS, float(P2_SYMBOL))):
                return True
        return False


    def winning_play(self, play):
        ttt = copy.deepcopy(self)
        ttt.play(play)
        if ttt.exists_winning_combination():
            return True
        else:
            return False

    def play(self, position):
        self.board[position] = self.turn.symbol
        self.turn = self.p1 if self.turn == self.p2 else self.p2

    def lines(self):
        result = []
        for i in range(ROWS):
            result.append(self.board[i,:])
            result.append(self.board[:,i])
        result.append(self.board.diagonal())
        result.append(np.fliplr(self.board).diagonal())
        return result

    def show(self):
        for i in range(0, ROWS):
            print('-------------')
            out = '| '
            for j in range(0, COLUMNS):
                out = out + str(self.board[i,j]) + ' | '
            print(out)
        print('-------------')

class Player:

    FILE_NAME = 'experience'
    WEIGHTS_FILE = 'weights'

    def __init__(self, symbol, name='terminator', eta=0.001):
        self.symbol = symbol
        self.plays = []
        self.name = name
        self.eta = eta
        self.values = np.zeros(6)

        self.load_weights()


    def set_tic_tac_toe(self, tic_tac_toe):
        self.tic_tac_toe = tic_tac_toe

    # ------------------------------------------------------------------------------------------------ #
    # The evaluation function will be w0x0 + w1x1 + w2x2 + w3x3 + w4x4 + w5x5                          #
    # x0 amount of lines where player 1 needs 0 pieces to win                                          #
    # x1 amount of lines where player 1 needs 1 pieces to win, and there's a free spot on that line    #
    # x2 amount of lines where player 1 needs 2 pieces to win, and there are 2 free spot on that line  #
    # ................................................................................................ #
    # x3 amount of lines where player 2 needs 0 pieces to win                                          #
    # x4 amount of lines where player 2 needs 1 pieces to win, and there's a free spot on that line    #
    # x5 amount of lines where player 2 needs 2 pieces to win, an0d there are 2 free spot on that line  #
    # ------------------------------------------------------------------------------------------------ #

    def choose_action(self):
        if positions := self.tic_tac_toe.avaliable_positions() :
            max_val = -float('inf')
            play = positions[0]
            for k in positions:
                tic_tac_toe = copy.deepcopy(self.tic_tac_toe)
                tic_tac_toe.play(k)
                lines = tic_tac_toe.lines()
                if self.evaluate(lines) >= max_val:
                    max_val = self.evaluate(lines)
                    play = k
            return play
        return False

    def evaluate(self, lines):
        self.calculate_values(lines)
        if self.values[ME_0_TO_WIN] > 0:
            return 100
        if self.values[OP_0_TO_WIN] > 0:
            return -100
        return np.dot(self.weights.T, self.values) # EVALUATE

    def calculate_values(self, lines):
        # check for conditions described and assign values to self.values
        self.values = np.zeros(6)
        self.values[ME_0_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 3, lines))
        self.values[ME_1_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 2 and collections.Counter(x)[float(EMPTY_CELL)]== 1, lines))
        self.values[ME_2_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 1 and collections.Counter(x)[float(EMPTY_CELL)]== 2, lines))
        self.values[OP_0_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 0 and collections.Counter(x)[float(EMPTY_CELL)]== 0, lines))
        self.values[OP_1_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 0 and collections.Counter(x)[float(EMPTY_CELL)]== 1, lines))
        self.values[OP_2_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 0 and collections.Counter(x)[float(EMPTY_CELL)]== 2, lines))

    def update_weights(self):
        for play in self.tic_tac_toe.avaliable_positions():
            player_copy = copy.deepcopy(self)
            tic_tac_training = copy.deepcopy(self.tic_tac_toe)

            tic_tac_training.play(position=play)

            v = self.evaluate(self.tic_tac_toe.lines())
            v_train = player_copy.evaluate(tic_tac_training.lines())
            for i in range(self.weights.size):
                self.weights[i] += self.eta*(v_train - v)*self.values[i]
            self.save_weights()

    # ------------------------------------------------------ #
    # --------------------- Load Data ---------------------- #
    # ------------------------------------------------------ #

    def load_weights(self):
        with open("{file}_{name}.pkl".format(file='weights', name=self.name), 'rb') as f:
            try:
                self.weights = pickle.load(f)
            except EOFError:
                self.weights = np.array([100,1,1,-100,-2,1], dtype='f')

    def save_weights(self):
        with open("{file}_{name}.pkl".format(file='weights', name=self.name), 'wb') as f:
            pickle.dump(self.weights, f)

class Judger:
    def __init__(self, player1, player2):
        self.p1 = player1
        self.p2 = player2
        self.ttt = TicTacToe(p1=self.p1, p2=self.p2)
        self.p1.set_tic_tac_toe(tic_tac_toe=self.ttt)
        self.p2.set_tic_tac_toe(tic_tac_toe=self.ttt)

    def reset(self):
        self.ttt = TicTacToe(p1=self.p1, p2=self.p2)
        self.p1.set_tic_tac_toe(tic_tac_toe=self.ttt)
        self.p2.set_tic_tac_toe(tic_tac_toe=self.ttt)

    def play(self):
        self.ttt.play(self.ttt.turn.choose_action())
        self.ttt.turns += 1

    def switch_sides(self):
        aux = self.p1         # Switch sides
        self.p1 = self.p2
        self.p2 = aux

    def play_against(self):
        if self.ttt.turn.name == 'human':
            self.ttt.turn.play()
        else:
            self.ttt.play(position=self.ttt.turn.choose_action())
        self.ttt.turns += 1

    def save_history(self, player1_wins, player2_wins, ties):
        with open("history.txt", "a+") as f:
            f.write("-----------------------------\n")
            f.write("player1 wins: {player1_wins} \n".format(player1_wins=player1_wins))
            f.write("player2 wins: {player2_wins} \n".format(player2_wins=player2_wins))
            f.write("ties: {ties} \n".format(ties=ties))

class Human:

    def __init__(self, name, symbol= P1_SYMBOL):
        self.name = name
        self.symbol = symbol

    def set_tic_tac_toe(self, tic_tac_toe):
        self.tic_tac_toe = tic_tac_toe

    def play(self):
        print('insert row')
        row = int(input())
        print('insert column')
        column = int(input())
        play = (row,column)
        self.tic_tac_toe.play(position=play)

class RandomPlayer:

    def __init__(self, name, symbol):
        self.name = name
        self.symbol = symbol

    def set_tic_tac_toe(self, tic_tac_toe):
        self.tic_tac_toe = tic_tac_toe

    def choose_action(self):
        possible_plays = self.tic_tac_toe.avaliable_positions()
        play = random.choice(possible_plays)
        return play

def train(rounds=500,eta=0.01, training=True, train_p1=True):
    player1 = Player(symbol= P1_SYMBOL, name='woodu', eta=eta)
    player2 = Player(symbol= P2_SYMBOL, name='player_two', eta=eta)
    player1_wins = 0
    ties = 0
    player2_wins = 0
    judger = Judger(player1=player1, player2=player2)
    for i in range(rounds):
        while not judger.ttt.has_ended():
            if training:
                if judger.ttt.turn.symbol != P1_SYMBOL or train_p1:
                    judger.ttt.turn.update_weights()
            judger.play()
        judger.ttt.show()
        print("round:", i + 1)
        if judger.ttt.who_won() is not None:
            print('winner: ', judger.ttt.who_won().symbol)
            if judger.ttt.who_won().symbol == P1_SYMBOL :
                player1_wins += 1
            else:
                player2_wins += 1
        else:
            print('tie')
            ties += 1

        print('weights player1:', player1.weights)
        print('weights player2:', player2.weights)
        judger.reset()

        judger.switch_sides()
    print('weights', player1.weights)
    print('weights', player2.weights)
    print('player 1 wins:', player1_wins)
    print('player 2 wins:', player2_wins)
    print('ties:', ties)
    player1.save_weights()
    player2.save_weights()

    judger.save_history(player1_wins=player1_wins, player2_wins=player2_wins, ties=ties)

def play_against():
    opponent = Player(symbol=P2_SYMBOL, name='woodu')
    player = Human(symbol=P1_SYMBOL, name='human')
    if randint(0,1) == 0:
        judger = Judger(player1=player, player2=opponent)
    else:
        judger = Judger(player1=opponent, player2=player)
    while not judger.ttt.has_ended():
        judger.play_against()
        judger.ttt.show()

    if judger.ttt.who_won() is not None:
        print('winner: ', judger.ttt.who_won().symbol)
    else:
        print('tie')

def random_train(rounds=500, training=True, eta=0.01):
    player1 = Player(symbol= P1_SYMBOL, name='woodu', eta=eta)
    player2 = RandomPlayer(symbol= P2_SYMBOL, name='randomator')
    player1_wins = 0
    ties = 0
    random_wins = 0
    judger = Judger(player1=player1, player2=player2)
    for i in range(rounds):
        while not judger.ttt.has_ended():
            if judger.ttt.turn.name == player1.name and training:
                judger.ttt.turn.update_weights()
            judger.play()
        judger.ttt.show()
        print("round:", i)
        print('weights player1:', player1.weights)
        if judger.ttt.who_won() is not None:
            print('winner: ', judger.ttt.who_won().symbol)
            if judger.ttt.who_won().symbol == P1_SYMBOL :
                player1_wins += 1
            if judger.ttt.who_won().symbol == P2_SYMBOL :
                random_wins += 1
        else:
            print('tie')
            ties += 1
        judger.reset()

        judger.switch_sides()
    print('player 1 wins:', player1_wins)
    print('random wins:', random_wins)
    print('ties:', ties)
    print('accuracy:', (player1_wins+ties)/rounds)
    player1.save_weights()

    judger.save_history(player1_wins=player1_wins, player2_wins=random_wins, ties=ties)
