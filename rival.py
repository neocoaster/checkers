import numpy as np

class Rival:

    def __init__(self):
        self.weights = np.zeros(6)
        self.values = np.zeros(6)
        self.trained_values = np.array([])

    def eval_target_function(self):
        calculate_weights(self)
        calculate_values(self)
        value(self)

    def calculate_values(self):
        self.values[0] = pieces_on_board(self, "black")
        self.values[1] = pieces_on_board(self, "red")
        self.values[2] = kings_on_board(self, "black")
        self.values[3] = kings_on_board(self, "red")
        self.values[4] = threatened_by(self, "red")
        self.values[5] = threatened_by(self, "black")

    def calculate_weights(self):
        NotImplemented

    def calculate_train_values(self):
        NotImplemented

    def value(self):
        np.dot(self.weights, self.values)

    def pieces_on_board(self, color):
        # This method should return the amount of color pieces on board
        NotImplemented

    def kings_on_board(self, color):
        # This method should return the amount of color kings on board
        NotImplemented

    def threatened_by(self, color):
        # This method should return the amount of threats by color player
        NotImplemented

    def next_boards(self):
        # This method should return the posible next boards for the current board status
        NotImplemented