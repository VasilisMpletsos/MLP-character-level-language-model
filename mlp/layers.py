from collections.abc import Iterator
from xmlrpc.client import boolean

import torch
from torch import Tensor


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class Linear:
    def __init__(self, input_neurons: int, output_neurons: int, bias: boolean = True):
        self.weights = torch.randn((input_neurons, output_neurons)) / input_neurons**0.5
        self.bias = torch.zeros(output_neurons) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        x = torch.matmul(x, self.weights)
        if self.bias is not None:
            x = x + self.bias
        self.out = x
        return self.out

    def parameters(self) -> list[Tensor]:
        return [self.weights] + ([self.bias] if self.bias is not None else [])

    def __repr__(self):
        weight_size = "("
        for neuron_size in self.weights.shape:
            weight_size += f"{neuron_size},"
        weight_size = weight_size[:-1] + ")"
        return f"Linear(weights={weight_size}, bias={'True' if self.bias is not None else 'False'})"


class BatchNormalization1D:
    def __init__(self, features: int, epsilon: float = 1e-5, momentum: float = 0.1):
        self.epsilon = torch.tensor(epsilon, requires_grad=False)
        self.momentum = torch.tensor(momentum, requires_grad=False)

        self.gamma = torch.ones(features)
        self.bias = torch.zeros(features)

        self.training = True
        self.mean = torch.zeros(features)
        self.std = torch.ones(features)

    def __call__(self, x: Tensor):
        if self.training:
            xmean = x.mean(0, keepdim=True)
            xvar = x.var(0, keepdim=True, unbiased=True)
        else:
            xmean = self.mean
            xvar = self.std

        self.out = (
            (x - xmean) / torch.sqrt(xvar + self.epsilon)
        ) * self.gamma + self.bias

        if self.training:
            with torch.no_grad():
                self.mean = (1 - self.momentum) * self.mean + self.momentum * xmean
                self.std = (1 - self.momentum) * self.std + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.bias]

    def __repr__(self):
        return f"BatchNormalization1D({self.epsilon=},{self.momentum=})"


class Tanh:
    def __init__(self):
        pass

    def __call__(self, x: Tensor):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, unique_indices: int, output_size: int):
        self.embedding = torch.randn((unique_indices, output_size))

    def __call__(self, x: Tensor):
        self.out = self.embedding[x]
        return self.out

    def parameters(self):
        return [self.embedding]


class WavenetFlatten:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(dim=1)
        self.out = x
        return self.out

    def parameters(self):
        return []
