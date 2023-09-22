import numpy as np
from numpy.linalg import inv


class LinearRegression(object):
    def __init__(self, number_of_features: int):
        self.w0 = 0
        self.w = np.zeros(number_of_features)

    def give_prediction(self, xi):
        return self.w0 + self.w.dot(xi)

    def give_predictions(self, x):
        w = np.array([self.w0] + self.w.tolist())  # check if you can make it better
        ones = np.ones(len(x))
        x = np.column_stack([ones, x])
        return x.dot(w)

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
