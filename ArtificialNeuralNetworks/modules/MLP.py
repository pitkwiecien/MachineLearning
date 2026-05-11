from typing import Callable
import numpy as np
import numpy.typing as npt
from .Layer import Layer
from .ActivationFunction import ActivationFunction


class MLP:
    def __init__(self, layer_sizes: list[int], activations: list[ActivationFunction], round_output: bool = False):
        if len(layer_sizes) - 1 != len(activations):
            raise ValueError("activations must match the number of layers")
        self.round_output = round_output

        self.layers: list[Layer] = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                Layer(
                    input_size=layer_sizes[i],
                    output_size=layer_sizes[i + 1],
                    activation=activations[i]
                )
            )

    def forward(self, input_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, loss_gradient: npt.NDArray[np.float64], learning_rate: float):
        gradient = loss_gradient

        for layer in reversed(self.layers):
            assert gradient is not None
            gradient = layer.backward(gradient, learning_rate)

    def predict(self, input_data: npt.NDArray[np.float64], round_output: bool | None = None) -> npt.NDArray[np.float64]:
        """Run forward pass and optionally round outputs.

        Args:
            input_data: input array
            round_output: override instance flag; if None, use `self.round_output`.

        Returns:
            Network outputs as an `np.ndarray` (float64). If rounded, values
            are rounded to nearest integer and clamped to [0, 5].
        """
        output = self.forward(input_data)
        do_round = self.round_output if round_output is None else round_output

        if do_round:
            rounded = np.round(output)
            # clamp to expected quality range [0, 5]
            rounded = np.clip(rounded, 0, 5)
            return rounded.astype(np.float64)

        return output

    def train_step(
        self,
        input_data: npt.NDArray[np.float64],
        target: npt.NDArray[np.float64],
        loss_function: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], float],
        loss_derivative: Callable[[npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        learning_rate: float
    ) -> float:
        output = self.forward(input_data)
        loss = loss_function(output, target)
        loss_grad = loss_derivative(output, target)
        self.backward(loss_grad, learning_rate)
        return loss
