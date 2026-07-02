# Examples

## XOR Problem

Binary classification of XOR inputs.

```python
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

model = Sequential()
model.add(Dense(2, 8))
model.add(ReLU())
model.add(Dense(8, 1))
model.add(Sigmoid())
model.compile(loss=BinaryCrossEntropy(), optimizer=Adam(), metric=Accuracy())
model.fit(X, y, epochs=1000)
```

## Iris Classification

Multi-class classification.

```bash
python examples/iris_classification.py
```

## MNIST Digit Classification

CNN for MNIST digits.

```bash
python examples/mnist_classifier.py
```

## Serialization

Save/load model workflow.

```python
model.save('model.krx')
loaded = load_model('model.krx')
```