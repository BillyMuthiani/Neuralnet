import numpy as np


def save(model, filename):

    params = {}

    idx = 0

    for layer in model.layers:

        if hasattr(layer, "weights"):

            params[f"w{idx}"] = layer.weights
            params[f"b{idx}"] = layer.biases

            idx += 1

    np.savez(filename, **params)


def load(model, filename):

    data = np.load(filename)

    idx = 0

    for layer in model.layers:

        if hasattr(layer, "weights"):

            layer.weights = data[f"w{idx}"]
            layer.biases = data[f"b{idx}"]

            idx += 1