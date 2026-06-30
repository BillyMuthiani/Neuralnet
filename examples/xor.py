import numpy as np

from Neuralnet.activations import ReLU, Sigmoid
from Neuralnet.layers import Dense
from Neuralnet.losses import BinaryCrossEntropy
from Neuralnet.metrics import Accuracy
from Neuralnet.model import Sequential
from Neuralnet.optimizers import SGD

# XOR dataset
x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0]).reshape(-1, 1)

print(f"Input shape: {x.shape}")
print(f"Target shape: {y.shape}")

# Build model
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

# Train
model.fit(x, y, epochs=10000)

# Evaluate
predictions = model.predict(x)
print(f"\nPredictions:\n{predictions}")
print(f"Targets:\n{y}")
