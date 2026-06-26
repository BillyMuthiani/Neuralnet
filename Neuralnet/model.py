import warnings

import numpy as np

from Neuralnet.exceptions import NotCompiledError


class History:

    def __init__(self):
        self.loss = []
        self.accuracy = []

    def __getitem__(self, key):
        if key == "loss":
            return self.loss
        elif key == "accuracy":
            return self.accuracy
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key == "loss":
            self.loss = value
        elif key == "accuracy":
            self.accuracy = value
        else:
            raise KeyError(key)

    def items(self):
        return [("loss", self.loss), ("accuracy", self.accuracy)]


class Sequential:

    def __init__(self):

        self.layers = []
        self.optimizer = None
        self.loss_function = None
        self.metric = None

    def add(self, layer):

        self.layers.append(layer)

    def forward(self, X):

        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, dvalues):

        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)

    def predict(self, X):

        return self.forward(X)

    def compile(
        self,
        loss,
        optimizer,
        metric=None
    ):

        self.loss_function = loss
        self.optimizer = optimizer
        self.metric = metric

    def save(self, filename):
        """Save model weights to a file."""
        from Neuralnet.serialization import save
        save(self, filename)

    def load(self, filename):
        """Load model weights from a file."""
        from Neuralnet.serialization import load
        load(self, filename)

    def fit(
        self,
        X,
        y,
        epochs=1000,
        loss=None,
        optimizer=None,
        metric=None
    ):

        if loss is not None or optimizer is not None or metric is not None:
            warnings.warn(
                "Passing loss, optimizer, or metric to fit() is deprecated. "
                "Use compile() to configure the model before calling fit().",
                DeprecationWarning,
                stacklevel=2
            )
            if loss is not None:
                self.loss_function = loss
            if optimizer is not None:
                self.optimizer = optimizer
            if metric is not None:
                self.metric = metric

        if self.loss_function is None:
            raise NotCompiledError(
                "Compile the model before calling fit()."
            )

        history = History()

        for epoch in range(epochs):

            predictions = self.forward(X)

            loss_value = self.loss_function.forward(
                y,
                predictions
            )

            history.loss.append(loss_value)

            dloss = self.loss_function.backward(
                y,
                predictions
            )

            self.backward(dloss)

            for layer in self.layers:
                self.optimizer.update(layer)

            if self.metric:

                score = self.metric.calculate(
                    y,
                    predictions
                )

                history.accuracy.append(score)

                if epoch % 100 == 0:
                    print(
                        f"Epoch {epoch} "
                        f"Loss: {loss_value:.6f} "
                        f"Accuracy: {score:.4f}"
                    )

            else:

                if epoch % 100 == 0:
                    print(
                        f"Epoch {epoch} "
                        f"Loss: {loss_value:.6f}"
                    )

        return history