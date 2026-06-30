import numpy as np

from Neuralnet.activations import Sigmoid, Tanh
from Neuralnet.layers import Dense
from Neuralnet.losses import BinaryCrossEntropy
from Neuralnet.metrics import Accuracy
from Neuralnet.model import Sequential
from Neuralnet.optimizers import Adam

x = np.array([
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

model = Sequential()

model.add(Dense(2, 8))
model.add(Tanh())

model.add(Dense(8, 1))
model.add(Sigmoid())

model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
    metric=Accuracy()
)

model.fit(
    x,
    y,
    epochs=5000
)

predictions = model.predict(x)

print(predictions)
