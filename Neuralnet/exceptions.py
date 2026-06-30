class NeuralnetError(Exception):
    """Base exception for all Neuralnet errors."""
    pass


class ConfigurationError(NeuralnetError):
    """Raised when model configuration is invalid."""
    pass


class NotCompiledError(ConfigurationError):
    """Raised when fit() is called before compile()."""
    pass


class ShapeError(NeuralnetError):
    """Raised when tensor shapes are incompatible."""
    pass


class OptimizerError(NeuralnetError):
    """Raised when optimizer operations fail."""
    pass
