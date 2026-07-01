# Training

Guide to training neural networks with Kronyx.

## Basic Training

```python
history = model.fit(X, y, epochs=100)
```

## Training Parameters

- `epochs`: Number of training iterations (default: 1000)
- `batch_size`: Samples per gradient update (default: all)
- `shuffle`: Shuffle training data each epoch (default: True)
- `validation_data`: Tuple (X_val, y_val) for validation
- `callbacks`: List of callback instances
- `verbose`: 0=silent, 1=progress

## Training History

```python
# Access metrics
history.loss
history.accuracy
history.val_loss
history.val_accuracy

# Export to CSV
history.to_csv('training_log.csv')

# View summary
print(history.summary())
```

## Validation

```python
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Val Loss: {loss:.4f}, Val Accuracy: {accuracy*100:.1f}%")
```