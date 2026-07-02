"""History to_dataframe demonstration.

Shows how to convert training history to pandas DataFrame for analysis.
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

history = model.fit(x_train, y_train, epochs=50, verbose=0)

print("=== History as DataFrame ===")
try:
    df = history.to_dataframe()
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn names:", list(df.columns))
    print("\nLoss statistics:")
    print(f"  Initial: {df['loss'].iloc[0]:.4f}")
    print(f"  Final: {df['loss'].iloc[-1]:.4f}")
    print(f"  Min: {df['loss'].min():.4f}")
except ImportError as e:
    print(f"Skipping DataFrame demo: {e}")
