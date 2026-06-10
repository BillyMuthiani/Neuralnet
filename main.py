import numpy as np

from Neuralnet.layers import Dense
from Neuralnet.activations import ReLU
from Neuralnet.activations import Sigmoid
from Neuralnet.losses import MSE
from Neuralnet.optimizers import SGD


dense1 = Dense(2, 4)

relu1 = ReLU()

dense2 = Dense(4, 1)

sigmoid1 = Sigmoid()

loss_function = MSE()

optimizer = SGD(0.1)


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


for epoch in range(5000):

    dense1_out = dense1.forward(X)

    relu_out = relu1.forward(
        dense1_out
    )

    dense2_out = dense2.forward(
        relu_out
    )

    predictions = sigmoid1.forward(
        dense2_out
    )

    loss = loss_function.forward(
        y,
        predictions
    )

    dloss = loss_function.backward(
        y,
        predictions
    )

    dsigmoid = sigmoid1.backward(
        dloss
    )

    dense2.backward(
        dsigmoid
    )

    drelu = relu1.backward(
        dense2.dinputs
    )

    dense1.backward(
        drelu
    )

    optimizer.update(dense1)

    optimizer.update(dense2)

    if epoch % 500 == 0:

        print(
            f"Epoch {epoch} Loss: {loss:.6f}"
        )
    dense1_out = dense1.forward(X)

relu_out = relu1.forward(
    dense1_out
)

dense2_out = dense2.forward(
    relu_out
)

predictions = sigmoid1.forward(
    dense2_out
)

print(predictions)