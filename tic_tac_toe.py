# Python 3.8
import numpy as np
import pickle
import copy

P1_SYMBOL = "❌"
P2_SYMBOL = "0️⃣"
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
        calculate_board_hash(self)

    def calculate_board_hash(self):
        self.board_hash = hash(str(self.board))

    def has_ended(self):
        if winner := winner(self) is not None:
            self.winner = winner
            self.reward(self.p1, self.p2) if winner == self.p1 else self.reward(self.p2, self.p1)
            return True
        elif self.turns == 9:
            return True
        else:
            return False

    def reward(self, winner, loser):
        winner.reward(1)
        loser.reward(0)

    def winner(self):
        if self.exists_winning_combination(self):
            return last_player_to_play(self)
        else:
            return None

    def last_player_to_play(self):
        if (self.turns % 2 == 0):
            return p2
        else:
            return p1

    def avaliable_positions(self, board=self.board):
        positions = []
        for i in range(ROWS):
            for j in range(COLUMNS):
                if board[i,j] == 0:
                    positions.append((i,j))
        return positions

    def exists_winning_comination(self):

        # Check for Columns combinations
        for i in range(ROWS):
            if (self.board[i,:] == np.full((COLUMNS), P1_SYMBOL)).all() or (self.board[i,:] == np.full((COLUMNS), P2_SYMBOL)).all():
                return True

        # Check for Rows combinations
        for j in range(COLUMNS):
            if (self.board[:,j] == np.full((ROWS), P1_SYMBOL)).all() or (self.board[:,j] == np.full((ROWS), P2_SYMBOL)).all():
                return True

        # Check for Diagonals
        for i in range(COLUMNS):
            if (self.board[i, i] == P1_SYMBOL):
                break
            if i == COLUMNS - 1:
                return True

        for i in range(COLUMNS):
            if (self.board[i, i] == P2_SYMBOL):
                break
            if i == COLUMNS - 1:
                return True

        for i in range(COLUMNS):
            if (self.board[COLUMNS - i, i] == P1_SYMBOL):
                break
            if i == COLUMNS - 1:
                return True

        for i in range(COLUMNS):
            if (self.board[COLUMNS - i, i] == P2_SYMBOL):
                break
            if i == COLUMNS - 1:
                return True

        return False

    def play(self, position):
        self.board[position] = self.turn.symbol
        self.turn = p1 if self.turn == self.p2 else p1


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
                out += self.board[i,j] + ' | '
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
        load_experience(self)
        load_weights(self)

    def set_tic_tac_toe(self, tic_tac_toe):
        self.tic_tac_toe = tic_tac_toe

    # Reinforcement learning ----------------------------------------------------BEGIN
    def make_action(self, positions, board):
        if np.random.uniform(0, 1) >= self.banana_rate:
            max_val = -float('inf')
            for k in range(positions):
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
        for key in reversed(self.plays):
            if self.experience.get(key) is None:
                self.experience[key] = 0

            # Here we should update the values with the evaluations
            self.experience[key] += self.weight * (value - self.experience[key])

        save_experience(self)
    # Reinforcement learning ----------------------------------------------------END


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

    ME_0_TO_WIN = 0
    ME_1_TO_WIN = 1
    ME_2_TO_WIN = 2
    OP_0_TO_WIN = 3
    OP_1_TO_WIN = 4
    OP_2_TO_WIN = 5

    def choose_action(self):
        positions = self.tic_tac_toe.avaliable_positions()
        for k in range(positions):
            tic_tac_toe = copy.deepcopy(self.tic_tac_toe)
            tic_tac_toe.play(k)
            lines = tic_tac_toe.lines()
            if value := evaluate(lines) >= max_val:
                max_val = value
                play = k
        return play

    def evaluate(self, lines):
        calculate_values(lines)
        if (self.values[ME_0_TO_WIN] > 0):           # If the plaher has 3 in a line
            return 100                               # WIN
        elif (self.values[OP_0_TO_WIN] > 0):         # If the opponent player has 3 in a line
            return -100                              # LOSE
        else:
            return np.dot(self.weights, self.values) # EVALUATE

    def calculate_values(self, lines):
        # check for conditions described and assign values to self.values
        self.values = np.zeros(6)                                                   # RESETS THE VALUES OF THE RANDOM DISCRETE VARIABLES
        self.values[ME_0_TO_WIN] = sum(map(lambda x : x.count(self.symbol) == 3, lines))
        self.values[ME_1_TO_WIN] = sum(map(lambda x : x.count(self.symbol) == 2 and x.count(EMPTY_CELL) == 1, lines))
        self.values[ME_2_TO_WIN] = sum(map(lambda x : x.count(self.symbol) == 1 and x.count(EMPTY_CELL) == 2, lines))
        self.values[OP_0_TO_WIN] = sum(map(lambda x : x.count(self.symbol) == 0 and x.count(EMPTY_CELL) == 0, lines))
        self.values[OP_1_TO_WIN] = sum(map(lambda x : x.count(self.symbol) == 0 and x.count(EMPTY_CELL) == 1, lines))
        self.values[OP_2_TO_WIN] = sum(map(lambda x : x.count(self.symbol) == 0 and x.count(EMPTY_CELL) == 2, lines))

    def update_weights(self):
        tic_tac_training = copy.deepcopy(self.tic_tac_toe).play(self.choose_action())
        v_train = self.evaluate(tic_tac_training)
        v = self.evaluate(self.tic_tac_toe.lines())
        for i in range(self.weights):
            self.weights[i] = self.weights[i] + self.eta*(v_train - v)*self.values[i]
        save_weights()



    # ------------------------------------------------------ #
    # --------------------- Load Data ---------------------- #
    # ------------------------------------------------------ #

    def load_experience(self):
        with open('{FILE_NAME}_{self.name}.pkl', 'rb') as f:
            self.experience = pickle.load(f)

    def save_experience(self):
        with open('{FILE_NAME}_{self.name}.pkl', 'wb') as f:
            pickle.dump(self.experience, f)

    def load_weights(self):
        with open('{WEIGHTS_FILE}_{self.name}.pkl', 'rb') as f:
            self.weights = pickle.load(f)

    def save_weights(self):
        with open('{WEIGHTS_FILE}_{self.name}.pkl', 'wb') as f:
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
        self.ttt.turn.play

def train(rounds=200):
    player1 = Player(symbol= P1_SYMBOL)
    player2 = Player(symbol= P2_SYMBOL)
    player1_wins = 0
    player2_wins = 0
    judger = Judger(player1=player1, player2=player2)
    for i in range(rounds):
        judger.play()
        print('round: ', i)
        if judger.ttt.has_ended():
            print('winner: ', judger.ttt.winner().symbol)
            if judger.ttt.winner().symbol == P1_SYMBOL :
                player1_wins += 1
            if judger.ttt.winner().symbol == P2_SYMBOL :
                player2_wins += 1
            judger.ttt.winner()
            judger.reset()

    player1.save_experience()
    player1.save_weights()
    player2.save_experience()
    player2.save_weights()
