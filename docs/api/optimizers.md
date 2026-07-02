# Optimizers

## SGD

Stochastic Gradient Descent.

```python
from kronyx import SGD

optimizer = SGD(learning_rate=0.01)
```

## Adam

Adam optimizer with momentum.

```python
from kronyx import Adam

optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
```