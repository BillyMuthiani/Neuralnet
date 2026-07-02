"""Model inspection demonstration.

Shows how to use model.summary(), model.count_params(), and model.visualize()
to understand your model architecture and parameters.
"""
from kronyx import Dense, ReLU, Sequential, Softmax

model = Sequential()
model.add(Dense(10, 32))
model.add(ReLU())
model.add(Dense(32, 16))
model.add(ReLU())
model.add(Dense(16, 3))
model.add(Softmax())

print("=== Model Summary ===")
model.summary()

print("\n=== Parameter Count ===")
total = model.count_params()
print(f"Total trainable parameters: {total:,}")

print("\n=== Layer Summary (structured data) ===")
layers_info = model.layer_summary()
for layer in layers_info:
    print(f"  {layer['name']}: {layer['params']:,} params (trainable={layer['trainable']})")

print("\n=== ASCII Visualization ===")
model.visualize()

print("\n=== ASCII Visualization with Parameters ===")
model.visualize(show_params=True)
