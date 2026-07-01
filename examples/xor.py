"""XOR binary classification example.

Demonstrates a simple neural network solving the XOR problem with binary cross-entropy loss.
"""
import numpy as np

from kronyx import (
    SGD,
    Accuracy,
    BinaryCrossEntropy,
    Dense,
    ReLU,
    Sequential,
    Sigmoid,
)

x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0]).reshape(-1, 1)

print(f"Input shape: {x.shape}")
print(f"Target shape: {y.shape}")

model = Sequential()
model.add(Dense(2, 8))
model.add(ReLU())
model.add(Dense(8, 1))
model.add(Sigmoid())

model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=SGD(learning_rate=0.1),
    metric=Accuracy()
)

model.fit(x, y, epochs=10000)

predictions = model.predict(x)
print(f"\nPredictions:\n{predictions}")
print(f"Targets:\n{y}")
