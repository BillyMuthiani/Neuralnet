"""Conv2D convolutional neural network demonstration.

Demonstrates using Conv2D layers for processing image-like data.
"""
import numpy as np

from kronyx import (
    Accuracy,
    Adam,
    Conv2D,
    Dense,
    Flatten,
    ReLU,
    Sequential,
    Softmax,
    SoftmaxCategoricalCrossEntropy,
)

np.random.seed(42)

x_train = np.random.randn(100, 8, 8, 3)
y_train = np.random.randint(0, 3, 100)

x_test = np.random.randn(20, 8, 8, 3)
y_test = np.random.randint(0, 3, 20)

print(f'Input shape: {x_train.shape}')

model = Sequential()

model.add(Conv2D(filters=8, kernel_size=3, padding='same'))
model.add(ReLU())
model.add(Conv2D(filters=16, kernel_size=3, padding='same', stride=2))
model.add(ReLU())
model.add(Flatten())
model.add(Dense(256, 3))
model.add(Softmax())

model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
    metric=Accuracy()
)

model.fit(
    x_train,
    y_train,
    epochs=50,
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
