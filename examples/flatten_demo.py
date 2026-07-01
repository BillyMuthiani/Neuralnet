"""Flatten layer demonstration with multi-dimensional input.

Demonstrates handling 2D input data (e.g., images) using the Flatten layer.
"""
import numpy as np

from kronyx import (
    Accuracy,
    Adam,
    Dense,
    Flatten,
    ReLU,
    Sequential,
    Softmax,
    SoftmaxCategoricalCrossEntropy,
)

np.random.seed(42)

x_train = np.random.randn(100, 2, 2)
y_train = np.random.randint(0, 3, 100)

x_test = np.random.randn(20, 2, 2)
y_test = np.random.randint(0, 3, 20)

print(f'Input shape: {x_train.shape}')

model = Sequential()

model.add(Flatten())
model.add(Dense(4, 8))
model.add(ReLU())
model.add(Dense(8, 3))
model.add(Softmax())

model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.01),
    metric=Accuracy()
)

model.fit(
    x_train,
    y_train,
    epochs=100,
)

predictions = model.predict(x_test)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

accuracy = np.mean(
    predicted_classes == y_test
)

print(
    f"Test Accuracy: {accuracy:.4f}"
)

print(f'Flatten shape cached: {model.layers[0].input_shape}')
