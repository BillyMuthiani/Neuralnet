# Kronyx

A lightweight deep learning framework built from first principles using NumPy. Designed for education, research, and production use with a clean Keras-like API.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/Kronyx/kronyx/actions/workflows/tests.yml/badge.svg)](https://github.com/Kronyx/kronyx/actions/workflows/tests.yml)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://mypy-lang.org/)

## Features

- **Layers**: Dense, Conv2D, BatchNormalization, Dropout, Flatten
- **Activations**: ReLU, Sigmoid, Tanh, Softmax
- **Optimizers**: SGD, Adam
- **Callbacks**: EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
- **Serialization**: Complete model save/load with `.krx` format
- **NumPy backend**: Pure NumPy implementation, no external ML dependencies
- **Type safe**: Full mypy type annotations

## Installation

```bash
pip install kronyx
```

For development:

```bash
git clone https://github.com/Kronyx/kronyx.git
cd kronyx
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from kronyx import Sequential, Dense, ReLU, Sigmoid, BinaryCrossEntropy, Adam, Accuracy

# XOR problem - binary classification
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(2, 8))  # input_size=2, output_size=8
model.add(ReLU())
model.add(Dense(8, 1))
model.add(Sigmoid())

model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.1),
    metric=Accuracy()
)

model.fit(X, y, epochs=1000)
predictions = model.predict(X)
print(f"Accuracy: {(predictions.round() == y).mean():.2%}")
```

## Binary Classification Example

```python
import numpy as np
from kronyx import Sequential, Dense, ReLU, Sigmoid, BinaryCrossEntropy, Adam, Accuracy

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(2, 16))
model.add(ReLU())
model.add(Dense(16, 1))
model.add(Sigmoid())

model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.1),
    metric=Accuracy()
)

history = model.fit(X_train, y_train, epochs=500)
model.summary()
```

## Multi-class Classification Example

```python
import numpy as np
from kronyx import Sequential, Dense, ReLU, SoftmaxCategoricalCrossEntropy, Adam, Accuracy

# One-hot encoded labels
X = np.random.randn(100, 4)
y = np.eye(3)[np.random.randint(0, 3, 100)]

model = Sequential()
model.add(Dense(4, 32))
model.add(ReLU())
model.add(Dense(32, 3))
model.add(Softmax())

model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.01),
    metric=Accuracy()
)

model.fit(X, y, epochs=100)
```

## Convolutional Neural Network Example

```python
import numpy as np
from kronyx import Sequential, Conv2D, ReLU, Flatten, Dense, Softmax

# Simple image input (batch, height, width, channels)
X = np.random.randn(10, 8, 8, 1)
y = np.eye(2)[np.random.randint(0, 2, 10)]

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=3, padding='same'))
model.add(ReLU())
model.add(Flatten())
model.add(Dense(64, 2))
model.add(Softmax())

model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.01),
    metric=Accuracy()
)

model.fit(X, y, epochs=10)
```

## Saving and Loading Models

```python
# Save complete model with architecture, weights, and configuration
model.save('model.krx')

# Load complete model
loaded = load_model('model.krx')

# Continue training or make predictions
loaded.predict(X_test)
```

## Saving and Loading Weights

```python
# Save only trainable weights
model.save_weights('weights.npz')

# Create a new model with matching architecture
new_model = Sequential()
new_model.add(Dense(2, 8))
new_model.add(ReLU())
new_model.add(Dense(8, 1))
new_model.add(Sigmoid())

# Load weights into the new model
new_model.load_weights('weights.npz')
```

## Exporting JSON Architecture

```python
# Export architecture to JSON string
json_str = model.to_json()

# Create model from JSON (weights initialized randomly)
model = Sequential.from_json(json_str)
```

## .krx Archive Format

The `.krx` format is a zip archive containing:

```
model.krx
├── metadata.json      # Framework version, Python/numpy versions, timestamp
├── architecture.json  # Layer configuration, loss, optimizer settings
├── weights.npz        # Trainable weights and biases (numpy archive)
└── optimizer.npz      # Optional: optimizer state for resumable training
```

## Examples

- `examples/xor.py` - XOR problem with binary classification
- `examples/iris_classification.py` - Iris dataset multi-class classification
- `examples/iris_dropout.py` - Dropout regularization example
- `examples/iris_l2.py` - L2 weight regularization
- `examples/batchnorm_iris.py` - Batch normalization demonstration
- `examples/flatten_demo.py` - Multi-dimensional input handling
- `examples/conv2d_demo.py` - Convolutional neural network example
- `examples/mnist_classifier.py` - MNIST digit classification

## Documentation

```
kronyx/
├── activations.py    # ReLU, Sigmoid, Tanh, Softmax
├── layers.py         # Dense, Conv2D, Flatten, Dropout, BatchNormalization
├── model.py          # Sequential, History
├── optimizers.py     # SGD, Adam
├── losses.py         # BinaryCrossEntropy, CategoricalCrossEntropy
├── metrics.py        # Accuracy
├── callbacks.py      # Callback base, EarlyStopping, etc.
├── regularizers.py   # L2
├── initializers.py   # he_normal, xavier_uniform, lecun_normal
├── serialization.py  # Save/load model (.krx format)
├── utils.py          # Utility functions
└── exceptions.py     # Error types
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features.

## License

MIT License - see [LICENSE](LICENSE)