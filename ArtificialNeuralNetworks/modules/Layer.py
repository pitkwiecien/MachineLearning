import numpy as np
import numpy.typing as npt
from typing import Optional
from .ActivationFunction import ActivationFunction


class Layer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationFunction
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.weights: npt.NDArray[np.float64] = (
            np.random.randn(output_size, input_size) * np.sqrt(2 / input_size)
        )
        self.biases: npt.NDArray[np.float64] = np.zeros((output_size, 1))
        self.input_data: Optional[npt.NDArray[np.float64]] = None
        self.linear_output: Optional[npt.NDArray[np.float64]] = None
        self.activation_output: Optional[npt.NDArray[np.float64]] = None
        self.activation = activation
        self.activation_function = activation.get_function()
        self.activation_derivative = activation.get_derivative()

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        self.input_data = input_data
        self.linear_output = self.weights @ input_data + self.biases
        self.activation_output = self.activation_function(self.linear_output)
        return self.activation_output

    def backward(self, gradient_from_next_layer: npt.NDArray[np.float64], learning_rate: float) -> npt.NDArray[np.float64]:
        assert self.linear_output is not None
        delta_pre_activation = gradient_from_next_layer * self.activation_derivative(self.linear_output)

        assert self.input_data is not None
        batch_size = self.input_data.shape[1]

        gradient_weights = (delta_pre_activation @ self.input_data.T) / batch_size
        gradient_biases = np.sum(delta_pre_activation, axis=1, keepdims=True) / batch_size
        gradient_input = self.weights.T @ delta_pre_activation

        self.weights -= learning_rate * gradient_weights
        self.biases -= learning_rate * gradient_biases

        return gradient_input
