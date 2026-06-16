import numpy as np


class Dense:

    def __init__(self, input_size, output_size):

        self.weights = (
            np.random.randn(input_size, output_size)
            * np.sqrt(2 / input_size)
        )

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