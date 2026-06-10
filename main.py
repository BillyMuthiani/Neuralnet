import numpy as np

from Neuralnet.layers import Dense
from Neuralnet.activations import ReLU
from Neuralnet.activations import Sigmoid

from Neuralnet.losses import MSE
from Neuralnet.optimizers import SGD

from Neuralnet.model import Sequential


X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


model = Sequential()

model.add(Dense(2,4))
model.add(ReLU())

model.add(Dense(4,1))
model.add(Sigmoid())


loss = MSE()

optimizer = SGD(
    learning_rate=0.1
)

model.fit(
    X,
    y,
    loss,
    optimizer,
    epochs=5000
)

predictions = model.predict(X)

print(predictions)