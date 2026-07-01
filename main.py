import numpy as np

from kronyx.activations import Sigmoid, Tanh
from kronyx.layers import Dense
from kronyx.losses import BinaryCrossEntropy
from kronyx.metrics import Accuracy
from kronyx.model import Sequential
from kronyx.optimizers import Adam

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
