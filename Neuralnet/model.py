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
        epochs=1000
    ):

        for epoch in range(epochs):

            predictions = self.forward(X)

            loss = loss_function.forward(
                y,
                predictions
            )

            dloss = loss_function.backward(
                y,
                predictions
            )

            self.backward(dloss)

            for layer in self.layers:

                optimizer.update(layer)

            if epoch % 100 == 0:

                print(
                    f"Epoch {epoch} "
                    f"Loss: {loss:.6f}"
                )