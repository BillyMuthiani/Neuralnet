import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from Neuralnet.activations import ReLU, Softmax
from Neuralnet.layers import Dense
from Neuralnet.losses import SoftmaxCategoricalCrossEntropy
from Neuralnet.metrics import Accuracy
from Neuralnet.model import Sequential
from Neuralnet.optimizers import Adam

digits = load_digits()

x = digits.data
y = digits.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

x_train = x_train / 16.0
x_test = x_test / 16.0

model = Sequential()

model.add(Dense(64, 128))
model.add(ReLU())

model.add(Dense(128, 64))
model.add(ReLU())

model.add(Dense(64, 10))
model.add(Softmax())

model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(
        learning_rate=0.001
    ),
    metric=Accuracy()
)

history = model.fit(
    x_train,
    y_train,
    epochs=1000
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
