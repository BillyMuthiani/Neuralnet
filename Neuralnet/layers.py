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

    def forward(self, X, training=True):

        if training:
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


class Dropout:
    """Dropout layer for regularization.

    Randomly zeros a fraction of input units during training to prevent overfitting.
    Uses inverted dropout scaling to maintain expected sums.
    """

    def __init__(self, rate):
        """Initialize the Dropout layer.

        Args:
            rate: Fraction of input units to drop (0 to 1). Must be in range [0, 1).

        Raises:
            ValueError: If rate is not in valid range.
        """
        if rate < 0 or rate >= 1:
            raise ValueError(
                f"rate must be in range [0, 1), got {rate}"
            )
        self.rate = rate
        self.mask = None
        self.input = None

    def forward(self, X, training=True):
        """Apply dropout to input.

        During training, randomly zeros activations with probability rate and scales
        remaining activations by 1/(1-rate). During inference, returns inputs unchanged.

        Args:
            X: Input array to apply dropout to.
            training: If True, apply dropout; if False, return inputs unchanged.

        Returns:
            Input with dropout applied during training, or original input during
            inference.
        """
        if training:
            self.input = X
            self.mask = np.random.rand(*X.shape) > self.rate
            return X * self.mask / (1 - self.rate)
        return X

    def backward(self, dvalues):
        """Backward pass applying cached dropout mask.

        Args:
            dvalues: Gradient values from the next layer.

        Returns:
            Gradient values with dropout mask applied.
        """
        return dvalues * self.mask / (1 - self.rate)