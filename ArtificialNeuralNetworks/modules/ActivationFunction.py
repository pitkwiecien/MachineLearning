from typing import Callable
from enum import Enum
import numpy.typing as npt
import numpy as np

class ActivationFunction(Enum):
    RELU = "RELU"
    SIGMOID = "SIGMOID"
    TANH = "TANH"
    LINEAR = "LINEAR"
    SOFTMAX = "SOFTMAX"

    @staticmethod
    def relu(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (z > 0).astype(np.float64)

    @staticmethod
    def sigmoid(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        s = ActivationFunction.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def tanh(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def linear(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return z

    @staticmethod
    def linear_derivative(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.ones_like(z)

    @staticmethod
    def softmax(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        shifted = z - np.max(z, axis=0, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=0, keepdims=True)

    @staticmethod
    def softmax_derivative(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        raise NotImplementedError("Softmax derivative is not used seperately")

    def get_function(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        return {
            ActivationFunction.RELU: ActivationFunction.relu,
            ActivationFunction.SIGMOID: ActivationFunction.sigmoid,
            ActivationFunction.TANH: ActivationFunction.tanh,
            ActivationFunction.LINEAR: ActivationFunction.linear,
            ActivationFunction.SOFTMAX: ActivationFunction.softmax,
        }[self]

    def get_derivative(self) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        return {
            ActivationFunction.RELU: ActivationFunction.relu_derivative,
            ActivationFunction.SIGMOID: ActivationFunction.sigmoid_derivative,
            ActivationFunction.TANH: ActivationFunction.tanh_derivative,
            ActivationFunction.LINEAR: ActivationFunction.linear_derivative,
            ActivationFunction.SOFTMAX: ActivationFunction.softmax_derivative,
        }[self]