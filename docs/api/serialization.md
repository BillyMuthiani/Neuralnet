# Serialization

## Save/Load Model

```python
# Save complete model
model.save('model.krx')

# Load model
loaded = load_model('model.krx')
```

## Save/Load Weights

```python
# Save weights only
model.save_weights('weights.npz')

# Load weights
model.load_weights('weights.npz')
```

## JSON Export/Import

```python
# Export to JSON
json_str = model.to_json()

# Create from JSON
model = Sequential.from_json(json_str)
```

## .krx Format

The `.krx` archive contains:

- `metadata.json` - Framework version, environment
- `architecture.json` - Layer configuration
- `weights.npz` - Trainable parameters
- `optimizer.npz` - Optional optimizer state