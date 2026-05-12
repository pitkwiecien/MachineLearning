import numpy as np
from .Layer import Layer
from .ActivationFunction import ActivationFunction


class MLP:

    def __init__(self, layer_sizes, activations, round_output=False):

        self.round_output = round_output

        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            for i in range(len(layer_sizes) - 1)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def train_step(self, x, y, loss_fn, loss_grad_fn, lr):

        out = self.forward(x)
        loss = loss_fn(out, y)

        grad = loss_grad_fn(out, y)
        self.backward(grad, lr)

        return loss