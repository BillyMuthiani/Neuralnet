# Callbacks

Callbacks allow custom behavior during training.

## EarlyStopping

Stop training when validation loss stops improving.

```python
from kronyx import EarlyStopping

callback = EarlyStopping(patience=10, min_delta=0.001)

model.fit(X, y, callbacks=[callback])
```

### Parameters

- `patience`: Epochs to wait before stopping
- `min_delta`: Minimum improvement threshold

## ModelCheckpoint

Save model after each epoch.

```python
from kronyx import ModelCheckpoint

callback = ModelCheckpoint(filepath='model.krx', save_best_only=True)

model.fit(X, y, callbacks=[callback])
```

### Parameters

- `filepath`: Where to save the model
- `save_best_only`: Only save if validation loss improved

## CSVLogger

Log training history to CSV.

```python
from kronyx import CSVLogger

callback = CSVLogger(filename='training_log.csv')

model.fit(X, y, callbacks=[callback])
```

## ReduceLROnPlateau

Reduce learning rate when metric plateaus.

```python
from kronyx import ReduceLROnPlateau

callback = ReduceLROnPlateau(factor=0.5, patience=5)

model.fit(X, y, callbacks=[callback])
```

### Parameters

- `factor`: Learning rate multiplier
- `patience`: Epochs to wait before reducing