# Python 3.8
import numpy as np

P1_SYMBOL = "❌"
P2_SYMBOL = "0️⃣"
ROWS = 3
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

    def avaliable_positions(self):
        positions = []
        for i in range(ROWS):
            for j in range(COLUMNS):
                if self.board[i,j] == 0:
                    positions.append((i,j))

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

class Player:
    def __init__(self, symbol, banana_rate, weight = 0.2):
        self.symbol = symbol
        # banana_rate means the % of times the player would take a random action, -- HE WENT BANANAS 🍌
        self.banana_rate = banana_rate
        self.plays = []
        self.weight = weight
        self.experience = {} # We could load the data from an external file and update it there

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
