# Layers

## Dense

Fully connected layer.

```python
from kronyx import Dense

layer = Dense(input_size=10, output_size=32)
```

### Parameters

- `input_size` (int): Number of input features
- `output_size` (int): Number of output neurons
- `initializer` (str): Weight initialization method
- `kernel_regularizer`: Optional L2 regularizer

## Conv2D

2D convolution layer.

```python
from kronyx import Conv2D

layer = Conv2D(filters=32, kernel_size=3, padding='same')
```

## Flatten

Flatten multi-dimensional input.

```python
from kronyx import Flatten

layer = Flatten()
```

## Dropout

Random regularization.

```python
from kronyx import Dropout

layer = Dropout(rate=0.5)
```

## BatchNormalization

Normalize inputs.

```python
from kronyx import BatchNormalization

layer = BatchNormalization(momentum=0.9)
```