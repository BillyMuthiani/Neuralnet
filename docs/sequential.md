# Sequential API

The `Sequential` class is the core model container for building neural networks.

## Creating a Model

```python
from kronyx import Sequential

model = Sequential()
```

## Adding Layers

```python
from kronyx import Dense, Conv2D, Flatten, Dropout

model = Sequential()
model.add(Dense(10, 32))  # input_size=10, output_size=32
model.add(Dense(32, 1))
```

## Compiling

```python
from kronyx import Adam, BinaryCrossEntropy, Accuracy

model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.01),
    metric=Accuracy()
)
```

### Arguments

- `loss`: Loss function instance (required)
- `optimizer`: Optimizer instance (required)
- `metric`: Optional metric for evaluation

## Training

```python
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    shuffle=True,
    validation_data=(X_val, y_val),
    verbose=1
)
```

### Return Value

Returns a `History` object with:
- `loss` - Training loss per epoch
- `accuracy` - Training accuracy per epoch
- `val_loss` - Validation loss (if provided)
- `val_accuracy` - Validation accuracy (if provided)

## Evaluation

```python
loss, accuracy = model.evaluate(X_test, y_test)
```

## Prediction

```python
predictions = model.predict(X, batch_size=None, verbose=0)
```

## Model Summary

```python
model.summary()
```

Output example:
```
============================================================
Layer                 Output Shape         Parameters   
============================================================
Dense0                  (None, 32)           352       
ReLU1                   (None, 32)           0         
Dense2                  (None, 1)            33        
============================================================
Trainable params: 385
============================================================
```