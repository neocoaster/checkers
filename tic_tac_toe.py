# Python 3.8
import numpy as np

P1_SYMBOL = "❌"
P2_SYMBOL = "0️⃣"
ROWS = 3
COLUMNS = 3

class TicTacToe(self, p1, p2):
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
            return True
        elif self.turns == 9:
            return True
        else:
            return False

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
