from collections.abc import Iterator
from xmlrpc.client import boolean

import torch
from torch import Tensor


class Linear:
    def __init__(self, input_neurons: int, output_neurons: int, bias: boolean = True):
        self.weights = torch.randn((input_neurons, output_neurons))
        self.bias = torch.zeros(output_neurons) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        x = torch.matmul(x, self.weights)
        if self.bias is not None:
            x = x + self.bias
        return x

    def parameters(self) -> list[Tensor]:
        return [self.weights] + ([self.bias] if self.bias is not None else [])

    def __repr__(self):
        weight_size = "("
        for neuron_size in self.weights.shape:
            weight_size += f"{neuron_size},"
        weight_size = weight_size[:-1] + ")"
        return f"Linear(weights={weight_size}, bias={'True' if self.bias is not None else 'False'})"
