import numpy as np


class MSE:
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true, y_pred):
        samples = len(y_true)

        return -2 * (y_true - y_pred) / samples