import numpy as np
from typing import Optional
from .ActivationFunction import ActivationFunction


class Layer:
    def __init__(self, input_size: int, output_size: int, activation: ActivationFunction):

        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.biases = np.zeros((1, output_size))

        self.input_data: Optional[np.ndarray] = None
        self.linear_output: Optional[np.ndarray] = None

        self.activation = activation
        self.act = activation.get_function()
        self.dact = activation.get_derivative()

    def forward(self, x):
        self.input_data = x
        self.linear_output = x @ self.weights + self.biases
        return self.act(self.linear_output)

    def backward(self, grad_out, lr):

        grad_act = grad_out * self.dact(self.linear_output)
        batch_size = self.input_data.shape[0]

        grad_w = (self.input_data.T @ grad_act) / batch_size
        grad_b = np.sum(grad_act, axis=0, keepdims=True) / batch_size
        grad_input = grad_act @ self.weights.T

        self.weights -= lr * grad_w
        self.biases -= lr * grad_b

        return grad_input