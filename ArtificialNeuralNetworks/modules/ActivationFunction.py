from enum import Enum
import numpy as np


class ActivationFunction(Enum):
    RELU = "RELU"
    SIGMOID = "SIGMOID"
    TANH = "TANH"
    LINEAR = "LINEAR"
    SOFTMAX = "SOFTMAX"

    @staticmethod
    def from_name(name: str):
        return ActivationFunction[name.strip().upper()]

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(np.float64)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = ActivationFunction.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def tanh(z):
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def linear(z):
        return z

    @staticmethod
    def linear_derivative(z):
        return np.ones_like(z)

    @staticmethod
    def softmax(z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def softmax_derivative(z):
        return np.ones_like(z)

    def get_function(self):
        return {
            ActivationFunction.RELU: self.relu,
            ActivationFunction.SIGMOID: self.sigmoid,
            ActivationFunction.TANH: self.tanh,
            ActivationFunction.LINEAR: self.linear,
            ActivationFunction.SOFTMAX: self.softmax,
        }[self]

    def get_derivative(self):
        return {
            ActivationFunction.RELU: self.relu_derivative,
            ActivationFunction.SIGMOID: self.sigmoid_derivative,
            ActivationFunction.TANH: self.tanh_derivative,
            ActivationFunction.LINEAR: self.linear_derivative,
            ActivationFunction.SOFTMAX: self.softmax_derivative,
        }[self]