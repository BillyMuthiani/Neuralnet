import numpy as np


class Accuracy:

    def calculate(
        self,
        y_true,
        y_pred
    ):

        if len(y_pred.shape) > 1:

            predictions = np.argmax(
                y_pred,
                axis=1
            )

        else:

            predictions = (
                y_pred > 0.5
            ).astype(int)

        return np.mean(
            predictions == y_true
        )