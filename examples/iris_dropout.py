import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from kronyx import (
    Accuracy,
    Adam,
    Dense,
    Dropout,
    ReLU,
    Sequential,
    Softmax,
    SoftmaxCategoricalCrossEntropy,
)

# Load Iris Dataset
iris = load_iris()

x = iris.data
y = iris.target

print(x.shape)
print(y.shape)

# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# Normalize Features
x_train = (
    x_train - x_train.mean(axis=0)
) / x_train.std(axis=0)

x_test = (
    x_test - x_test.mean(axis=0)
) / x_test.std(axis=0)

# Build Model with Dropout Regularization
model = Sequential()

model.add(Dense(4, 16))
model.add(ReLU())
model.add(Dropout(0.3))

model.add(Dense(16, 3))
model.add(Softmax())

# Train
model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.01),
    metric=Accuracy()
)

model.fit(
    x_train,
    y_train,
    epochs=1000
)

# Evaluate
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
