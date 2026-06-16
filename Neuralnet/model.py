class Sequential:

    def __init__(self):

        self.layers = []

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

    def fit(
        self,
        X,
        y,
        loss_function,
        optimizer,
        epochs=1000,
        metric=None
    ):

        history = {
            "loss": []
        }

        if metric:
            history["accuracy"] = []

        for epoch in range(epochs):

            predictions = self.forward(X)

            loss = loss_function.forward(
                y,
                predictions
            )

            history["loss"].append(loss)

            dloss = loss_function.backward(
                y,
                predictions
            )

            self.backward(dloss)

            for layer in self.layers:
                optimizer.update(layer)

            if metric:

                score = metric.calculate(
                    y,
                    predictions
                )

                history["accuracy"].append(
                    score
                )

            print(
                f"Epoch {epoch} "
                f"Loss: {loss:.6f}"
            )

        return history
    