import numpy as np
from numpy.linalg import inv


class LinearRegressionModel(object):
    def __init__(self, number_of_features: int):
        self.w0 = 0
        self.w = np.zeros(number_of_features)

    def predict(self, x):
        return self.w0 + x.dot(self.w)

    def train(self, x, y):
        ones = np.ones(len(x))
        x = np.column_stack([ones, x])
        w = inv(x.T.dot(x)).dot(x.T).dot(y)
        self.w0 = w[0]
        self.w = w[1:]

    @staticmethod
    def rmse(y, y_pred):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)
