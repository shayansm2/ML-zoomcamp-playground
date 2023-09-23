import numpy as np
from numpy.linalg import inv

from week2.LinearRegressionModel import LinearRegressionModel


class RidgeRegressionModel(LinearRegressionModel):
    def __init__(self, number_of_features: int, regularization_factor: int = 0.01):
        super().__init__(number_of_features)
        self.r = regularization_factor

    def train(self, x, y):
        ones = np.ones(len(x))
        x = np.column_stack([ones, x])
        xtx = x.T.dot(x)
        reg = self.r * np.eye(xtx.shape[0])
        xtx += reg
        w = inv(xtx).dot(x.T).dot(y)
        self.w0 = w[0]
        self.w = w[1:]
