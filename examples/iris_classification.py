"""Iris dataset multi-class classification example.

Demonstrates training a neural network on the Iris dataset with softmax activation
and categorical cross-entropy loss for 3-class classification.
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from kronyx import Accuracy, Adam, Dense, ReLU, Sequential, Softmax, SoftmaxCategoricalCrossEntropy

iris = load_iris()

x = iris.data
y = iris.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)

x_train = (
    x_train - x_train.mean(axis=0)
) / x_train.std(axis=0)

x_test = (
    x_test - x_test.mean(axis=0)
) / x_test.std(axis=0)

model = Sequential()

model.add(Dense(4, 16))
model.add(ReLU())

model.add(Dense(16, 3))
model.add(Softmax())

model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
    metric=Accuracy()
)

model.fit(
    x_train,
    y_train,
    epochs=5000,
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
