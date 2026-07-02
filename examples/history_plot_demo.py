"""History plotting demonstration.

This example shows how to use history.plot() to visualize training metrics.
The plot() method creates informative charts of loss and accuracy curves.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from kronyx import Accuracy, Adam, Dense, ReLU, Sequential, Softmax, SoftmaxCategoricalCrossEntropy

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0)
x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)

model = Sequential()
model.add(Dense(4, 16))
model.add(ReLU())
model.add(Dense(16, 3))
model.add(Softmax())

model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.01),
    metric=Accuracy()
)

history = model.fit(x_train, y_train, epochs=200, verbose=0)

print("=== History Summary ===")
print(history.summary())

print("\n=== Plotting metrics (requires matplotlib) ===")
try:
    history.plot()
    print("Displaying plot window...")
except ImportError as e:
    print(f"Skipping plot: {e}")

print("\n=== Plotting only loss ===")
try:
    history.plot(metric='loss')
except ImportError as e:
    print(f"Skipping plot: {e}")
