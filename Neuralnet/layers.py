import numpy as np
from Neuralnet.initializers import he_normal, xavier_uniform, lecun_normal


class Dense:

    def __init__(self, input_size, output_size, initializer="he_normal"):

        init_func = {
            "he_normal": he_normal,
            "xavier_uniform": xavier_uniform,
            "lecun_normal": lecun_normal,
        }.get(initializer)

        if init_func is None:
            raise ValueError(
                f"Unknown initializer: {initializer}. "
                f"Available: he_normal, xavier_uniform, lecun_normal"
            )

        self.weights = init_func(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, X):

        self.input = X

        return np.dot(X, self.weights) + self.biases

    def backward(self, dvalues):

        self.dweights = np.dot(
            self.input.T,
            dvalues
        )

        self.dbiases = np.sum(
            dvalues,
            axis=0,
            keepdims=True
        )

        self.dinputs = np.dot(
            dvalues,
            self.weights.T
        )

        return self.dinputs