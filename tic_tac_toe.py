# Python 3.8
import numpy as np
import collections
import pickle
import copy

P1_SYMBOL = "1"
P2_SYMBOL = "2"
ROWS = 3
EMPTY_CELL = 0
COLUMNS = 3
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
        if winner:=self.who_win() is not None:
            self.winner = winner
            print('rewarding')
            self.reward(self.p1, self.p2) if winner == self.p1 else self.reward(self.p2, self.p1)
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

    def __init__(self, symbol, banana_rate=0.3, weight=0.2, name='terminator', eta=0.1):
        self.symbol = symbol
        # banana_rate means the % of times the player would take a random action, -- HE WENT BANANAS 🍌
        self.banana_rate = banana_rate
        self.plays = []
        self.weight = weight
        self.name = name
        self.eta = eta
        self.values = np.zeros(6)
        self.weights = np.zeros(6)
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
                if value := value(self, possible_board_hash) >= max_val:
                    max_val = value
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
            self.experience[i] += self.weights[i] * (value - self.experience[i])
        print(self.experience)
        self.save_experience()


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
        positions = self.tic_tac_toe.avaliable_positions()
        if positions :
            max_val=0
            play = positions[0]
            for k in positions:
                tic_tac_toe = copy.deepcopy(self.tic_tac_toe)
                tic_tac_toe.play(k)
                lines = tic_tac_toe.lines()
                if value := self.evaluate(lines) >= max_val:
                    max_val = value
                    play = k
            return play
        return False

    def evaluate(self, lines):
        ME_0_TO_WIN = 0
        ME_1_TO_WIN = 1
        ME_2_TO_WIN = 2
        OP_0_TO_WIN = 3
        OP_1_TO_WIN = 4
        OP_2_TO_WIN = 5
        self.calculate_values(lines)
        if (self.values[ME_0_TO_WIN] > 0):           # If the plaher has 3 in a line
            return 100                               # WIN
        elif (self.values[OP_0_TO_WIN] > 0):         # If the opponent player has 3 in a line
            return -100                              # LOSE
        else:
            return np.dot(self.weights, self.values) # EVALUATE

    def calculate_values(self, lines):
        # check for conditions described and assign values to self.values
        ME_0_TO_WIN = 0
        ME_1_TO_WIN = 1
        ME_2_TO_WIN = 2
        OP_0_TO_WIN = 3
        OP_1_TO_WIN = 4
        OP_2_TO_WIN = 5
        self.values = np.zeros(6)
        self.values[ME_0_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 3, lines))
        self.values[ME_1_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 2 and collections.Counter(x)[EMPTY_CELL]== 1, lines))
        self.values[ME_2_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 1 and collections.Counter(x)[EMPTY_CELL]== 2, lines))
        self.values[OP_0_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 0 and collections.Counter(x)[EMPTY_CELL]== 0, lines))
        self.values[OP_1_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 0 and collections.Counter(x)[EMPTY_CELL]== 1, lines))
        self.values[OP_2_TO_WIN] = sum(map(lambda x : collections.Counter(x)[float(self.symbol)] == 0 and collections.Counter(x)[EMPTY_CELL]== 2, lines))

    def update_weights(self):
        tic_tac_training = copy.deepcopy(self.tic_tac_toe)
        if self.choose_action():
            tic_tac_training.play(self.choose_action())
            if lines:=tic_tac_training.lines():
                v_train = self.evaluate(lines)
                v = self.evaluate(self.tic_tac_toe.lines())
                for i in range(self.weights.size):
                    self.weights[i] = self.weights[i] + self.eta*(v_train - v)*self.values[i]
                self.save_weights()



    # ------------------------------------------------------ #
    # --------------------- Load Data ---------------------- #
    # ------------------------------------------------------ #

    def load_experience(self):
        with open("{file}_{name}.pkl".format(file='experience', name=self.name), 'rb') as f:
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
                self.experience = np.zeros(6)

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
        judger.ttt.show()
        if judger.ttt.who_win() is not None:
            print('winner: ', judger.ttt.who_win().symbol)
            if judger.ttt.who_win().symbol == P1_SYMBOL :
                player1_wins += 1
            if judger.ttt.who_win().symbol == P2_SYMBOL :
                player2_wins += 1
        else:
            print('tie')
            ties += 1
        judger.reset()
    print('weights', player1.weights)
    print('weights', player2.weights)
    print('experience', player1.experience)
    print('experience', player2.experience)
    print('player 1 wins:', player1_wins)
    print('player 2 wins:', player2_wins)
    print('ties:', ties)

    player1.save_experience()
    player1.save_weights()
    player2.save_experience()
    player2.save_weights()


train()