# Quick Start

This guide will help you build your first neural network with Kronyx.

## Import

```python
import numpy as np
from kronyx import Sequential, Dense, ReLU, Sigmoid, BinaryCrossEntropy, Adam, Accuracy
```

## Create a Model

```python
model = Sequential()
model.add(Dense(2, 8))  # input_size=2, output_size=8
model.add(ReLU())
model.add(Dense(8, 1))
model.add(Sigmoid())
```

## Compile

```python
model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.1),
    metric=Accuracy()
)
```

## Prepare Data

```python
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])
```

## Train

```python
history = model.fit(X_train, y_train, epochs=1000)
```

## Evaluate

```python
predictions = model.predict(X_train)
accuracy = (predictions.round() == y_train).mean()
print(f"Accuracy: {accuracy:.2%}")
```

## Save Model

```python
model.save('model.krx')
```

## Load Model

```python
from kronyx import load_model
loaded = load_model('model.krx')
```