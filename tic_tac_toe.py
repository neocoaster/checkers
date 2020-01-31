# Python 3.8
import numpy as np
import collections
import time
import pickle
import copy
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
        self.calculate_board_hash()

    def calculate_board_hash(self):
        self.board_hash = hash(str(self.board))

    def has_ended(self):
        if self.who_win() is not None:
            self.winner = self.who_win()
            return True
        elif self.turns == 9:
            return True
        else:
            return False

    def reward(self, winner, loser):
        winner.reward(1)
        loser.reward(0)

    def who_win(self):
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

        # Check for Columns combinations
        for i in range(ROWS):
            if (self.board[i,:] == np.full(COLUMNS, float(P1_SYMBOL))).all() or (self.board[i,:] == np.full((COLUMNS), float(P2_SYMBOL))).all():
                return True

        # Check for Rows combinations
        for j in range(COLUMNS):
            if (self.board[:,j] == np.full(ROWS, float(P1_SYMBOL))).all() or (self.board[:,j] == np.full((ROWS), float(P2_SYMBOL))).all():
                return True

        # Check for Diagonals
        for i in range(COLUMNS):
            if (self.board[i, i] != float(P1_SYMBOL)):
                break
            if i == COLUMNS - 1:
                return True

        for i in range(COLUMNS):
            if (self.board[i, i] != float(P2_SYMBOL)):
                break
            if i == COLUMNS - 1:
                return True

        for i in range(COLUMNS):
            if (self.board[COLUMNS - i - 1, i] != float(P1_SYMBOL)):
                break
            if i == COLUMNS - 1:
                return True

        for i in range(COLUMNS):
            if (self.board[COLUMNS - i - 1, i] != float(P2_SYMBOL)):
                break
            if i == COLUMNS - 1:
                return True

        return False

    def winning_play(self, play):
        ttt = copy.deepcopy(self)
        ttt.play(play)
        if ttt.has_ended():
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

    def __init__(self, symbol, banana_rate=0.3, weight=0.2, name='terminator', eta=0.00015):
        self.symbol = symbol
        # banana_rate means the % of times the player would take a random action, -- HE WENT BANANAS \
        self.banana_rate = banana_rate
        self.plays = []
        self.weight = weight
        self.name = name
        self.eta = eta
        self.values = np.zeros(6)

        self.load_experience()
        self.load_weights()


    def set_tic_tac_toe(self, tic_tac_toe):
        self.tic_tac_toe = tic_tac_toe

    # Reinforcement learning ----------------------------------------------------BEGIN
    def make_action(self, positions, board):
        if np.random.uniform(0, 1) >= self.banana_rate:
            max_val = -float('inf')
            for k in positions:
                possible_board = board.copy()
                possible_board[k] = self.symbol
                possible_board_hash = hash(str(possible_board))
                if value(self, possible_board_hash) >= max_val:
                    max_val = value(self, possible_board_hash)
                    play = k
        else:
            play_index = np.random.choice(len(positions))
            play = positions[play_index]
        return play

    def value(self, board_hash):
        if value := self.experience.get(board_hash) is None:
            return 0
        else:
            return value

    def reward(self, value):
        for i in range(self.experience.size):
            self.experience[i] = self.weights[i] * (value - self.experience[i])
        self.save_experience()

    # Reinforcement learning ------------------------------------------------------END


    # ------------------------------------------------------------------------------------------------ #
    # The evaluation function will be w0x0 + w1x1 + w2x2 + w3x3 + w4x4 + w5x5                          #
    # x0 amount of lines where player 1 needs 0 pieces to win                                          #
    # x1 amount of lines where player 1 needs 1 pieces to win, and there's a free spot on that line    #
    # x2 amount of lines where player 1 needs 2 pieces to win, and there are 2 free spot on that line  #
    # ................................................................................................ #
    # x3 amount of lines where player 2 needs 0 pieces to win                                          #
    # x4 amount of lines where player 2 needs 1 pieces to win, and there's a free spot on that line    #
    # x5 amount of lines where player 2 needs 2 pieces to win, and there are 2 free spot on that line  #
    # ------------------------------------------------------------------------------------------------ #



    def choose_action(self):
        if positions:=self.tic_tac_toe.avaliable_positions() :
            max_val = -float('inf')
            play = positions[0]
            for k in positions:
                tic_tac_toe = copy.deepcopy(self.tic_tac_toe)
                tic_tac_toe.play(k)
                lines = tic_tac_toe.lines()
                if tic_tac_toe.winning_play(k):
                    return k
                if self.evaluate(lines) >= max_val:
                    max_val = self.evaluate(lines)
                    play = k
            return play
        return False

    def evaluate(self, lines):
        self.calculate_values(lines)
        if self.values[ME_0_TO_WIN] > 0:
            return 10000
        if self.values[OP_0_TO_WIN] > 0:
            return -10000
        return np.dot(self.weights, self.values) # EVALUATE

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
            tic_tac_training = player_copy.tic_tac_toe
            tic_tac_training.play(position=play)
            if lines:=tic_tac_training.lines():

                v = self.evaluate(self.tic_tac_toe.lines())
                v_train = self.evaluate(lines)
                for i in range(self.weights.size):
                    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    # print(self.eta*(v_train - v)*player_copy.values[i])
                    # print(self.eta)
                    # print(v_train)
                    # print(v)
                    # print((v_train - v))
                    # print(player_copy.values[i])
                    # print('--------------------------------')
                    self.weights[i] += self.eta*(v_train - v)*self.values[i]
                self.save_weights()



    # ------------------------------------------------------ #
    # --------------------- Load Data ---------------------- #
    # ------------------------------------------------------ #

    def load_experience(self):
        with open("{file}_{name}.pkl".format(file='experience', name='player_one'), 'rb') as f:
            try:
                self.experience = pickle.load(f)
            except EOFError:
                self.experience = np.zeros(6)

    def save_experience(self):
        with open("{file}_{name}.pkl".format(file='experience', name=self.name), 'wb') as f:
            pickle.dump(self.experience, f)

    def load_weights(self):
        with open("{file}_{name}.pkl".format(file='weights', name=self.name), 'rb') as f:
            try:
                self.weights = pickle.load(f)
            except EOFError:
                self.weights = np.array([0,0,0,0,0,0], dtype='f')

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

def train(rounds=500):
    player1 = Player(symbol= P1_SYMBOL, name='player_one')
    player2 = Player(symbol= P2_SYMBOL, name='player_two')
    player1_wins = 0
    ties = 0
    player2_wins = 0
    judger = Judger(player1=player1, player2=player2)
    for i in range(rounds):
        while not judger.ttt.has_ended():
            judger.play()
            player1.update_weights()
            player2.update_weights()
        player1.update_weights()
        player2.update_weights()
        judger.ttt.show()
        print("round:", i)
        if judger.ttt.who_win() is not None:
            print('winner: ', judger.ttt.who_win().symbol)
            if judger.ttt.who_win().symbol == P1_SYMBOL :
                player1_wins += 1
            if judger.ttt.who_win().symbol == P2_SYMBOL :
                player2_wins += 1
        else:
            print('tie')
            ties += 1

        print('weights player1:', player1.weights)
        print('weights player2:', player2.weights)
        judger.reset()

        aux = judger.p1         # Switch sides
        judger.p1 = judger.p2
        judger.p2 = aux
    print('weights', player1.weights)
    print('weights', player2.weights)
    print('player 1 wins:', player1_wins)
    print('player 2 wins:', player2_wins)
    print('ties:', ties)

    player1.save_experience()
    player1.save_weights()
    player2.save_experience()
    player2.save_weights()

    judger.save_history(player1_wins=player1_wins, player2_wins=player2_wins, ties=ties)

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

def play_against():
    opponent = Player(symbol=P2_SYMBOL, name='woodu')
    player = Human(symbol=P1_SYMBOL, name='human')
    judger = Judger(player1=player, player2=opponent)
    while not judger.ttt.has_ended():
        judger.ttt.show()
        judger.play_against()

    if judger.ttt.who_win() is not None:
        print('winner: ', judger.ttt.who_win().symbol)
    else:
        print('tie')


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


def random_train(rounds=500):
    player1 = Player(symbol= P1_SYMBOL, name='woodu')
    player2 = RandomPlayer(symbol= P2_SYMBOL, name='randomator')
    player1_wins = 0
    ties = 0
    random_wins = 0
    judger = Judger(player1=player1, player2=player2)
    for i in range(rounds):
        while not judger.ttt.has_ended():
            print('weights player1:', player1.weights)
            judger.play()
            player1.update_weights()
            print('weights player1:', player1.weights)
        judger.ttt.show()
        print("round:", i)
        print('weights player1:', player1.weights)
        if judger.ttt.who_win() is not None:
            print('winner: ', judger.ttt.who_win().symbol)
            if judger.ttt.who_win().symbol == P1_SYMBOL :
                player1_wins += 1
            if judger.ttt.who_win().symbol == P2_SYMBOL :
                random_wins += 1
        else:
            print('tie')
            ties += 1
        judger.reset()

        aux = judger.p1         # Switch sides
        judger.p1 = judger.p2
        judger.p2 = aux
    print('player 1 wins:', player1_wins)
    print('random wins:', random_wins)
    print('ties:', ties)

    player1.save_weights()

    judger.save_history(player1_wins=player1_wins, player2_wins=random_wins, ties=ties)
