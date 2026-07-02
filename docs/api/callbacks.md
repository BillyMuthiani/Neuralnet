# Callbacks

## EarlyStopping

Stop training early.

```python
from kronyx import EarlyStopping

callback = EarlyStopping(patience=10, min_delta=0.001)
```

## ModelCheckpoint

Save model checkpoints.

```python
from kronyx import ModelCheckpoint

callback = ModelCheckpoint(filepath='model.krx', save_best_only=True)
```

## CSVLogger

Log to CSV.

```python
from kronyx import CSVLogger

callback = CSVLogger(filename='training_log.csv')
```

## ReduceLROnPlateau

Reduce learning rate on plateau.

```python
from kronyx import ReduceLROnPlateau

callback = ReduceLROnPlateau(factor=0.5, patience=5)
```