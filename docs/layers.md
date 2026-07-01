# Layers

Kronyx provides several layer types for building neural networks.

## Dense

Fully connected layer.

```python
from kronyx import Dense

layer = Dense(input_size=10, output_size=32)
```

### Parameters

- `input_size`: Number of input features
- `output_size`: Number of output neurons
- `initializer`: Weight initialization method ("he_normal", "xavier_uniform", "lecun_normal")
- `kernel_regularizer`: Optional L2 regularizer instance

## Conv2D

2D convolutional layer for image inputs.

```python
from kronyx import Conv2D

layer = Conv2D(filters=32, kernel_size=3, padding='same')
```

### Parameters

- `filters`: Number of output filters
- `kernel_size`: Size of convolution kernel (int or tuple)
- `padding`: "valid" or "same"

## Flatten

Flattens multi-dimensional input to 1D.

```python
from kronyx import Flatten

layer = Flatten()
```

## Dropout

Regularization layer that randomly zeros inputs during training.

```python
from kronyx import Dropout

layer = Dropout(rate=0.5)
```

### Parameters

- `rate`: Fraction of inputs to drop (0 to 1)

## BatchNormalization

Normalizes layer inputs for stable training.

```python
from kronyx import BatchNormalization

layer = BatchNormalization(momentum=0.9)
```

### Parameters

- `momentum`: Momentum for running mean/variance updates