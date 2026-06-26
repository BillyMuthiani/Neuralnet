class Callback:

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class EarlyStopping(Callback):

    def __init__(
        self,
        monitor="val_loss",
        patience=10,
        min_delta=0.0,
        restore_best_weights=True
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.best = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return

        current = logs.get(self.monitor)

        if current is None:
            return

        if self.best is None:
            self.best = current
            self.best_weights = self._get_weights()
            self.wait = 0
        elif self._is_improvement(current):
            self.best = current
            self.best_weights = self._get_weights()
            self.wait = 0
        else:
            self.wait += 1

            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.restore_best_weights and self.best_weights is not None:
            self._set_weights(self.best_weights)

    def _get_weights(self):
        weights = []
        for layer in self.model.layers:
            if hasattr(layer, "weights"):
                weights.append((layer.weights.copy(), layer.biases.copy()))
        return weights

    def _set_weights(self, weights):
        for layer, weight_data in zip(
            [l for l in self.model.layers if hasattr(l, "weights")],
            weights
        ):
            layer.weights, layer.biases = weight_data[0].copy(), weight_data[1].copy()

    def _is_improvement(self, current):
        if self.monitor.startswith("val_"):
            return current < self.best - self.min_delta
        else:
            return current > self.best + self.min_delta