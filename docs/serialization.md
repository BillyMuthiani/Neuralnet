# Serialization

Save and load models for later use.

## Saving Models

```python
# Save complete model with architecture and weights
model.save('model.krx')
```

## Loading Models

```python
from kronyx import load_model

model = load_model('model.krx')
```

## Saving Weights Only

```python
# Save only trainable parameters
model.save_weights('weights.npz')

# Load into compatible architecture
model.load_weights('weights.npz')
```

## JSON Export

```python
# Export architecture to JSON
json_str = model.to_json()

# Create model from JSON
model = Sequential.from_json(json_str)
```

## .krx Archive Format

The `.krx` format is a zip archive containing:

```
model.krx
├── metadata.json      # Framework info and version
├── architecture.json  # Layer structure
├── weights.npz        # Trainable parameters
└── optimizer.npz      # Optional optimizer state
```

All serialization errors raise `SerializationError`.