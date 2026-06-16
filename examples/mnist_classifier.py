from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from Neuralnet.model import Sequential
from Neuralnet.layers import Dense
from Neuralnet.activations import ReLU, Softmax
from Neuralnet.losses import SoftmaxCategoricalCrossEntropy
from Neuralnet.optimizers import Adam
from Neuralnet.metrics import Accuracy
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()

X = digits.data
y = digits.target

print(X.shape)
print(y.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = X_train / 16.0
X_test = X_test / 16.0

model = Sequential()

model.add(Dense(64, 128))
model.add(ReLU())

model.add(Dense(128, 64))
model.add(ReLU())

model.add(Dense(64, 10))
model.add(Softmax())

model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
    metric=Accuracy()
)

history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    
)

predictions = model.predict(X_test)

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



index = 0

plt.imshow(
    X_test[index].reshape(8, 8),
    cmap="gray"
)

plt.title(
    f"Predicted: {predicted_classes[index]}"
)

plt.show()