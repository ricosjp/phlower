from collections.abc import Callable
from typing import TypeVar

import torch

from phlower._base import IPhlowerTensor
from phlower.utils.enums import ActivationType
from phlower.utils.exceptions import PhlowerInvalidActivationError


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def leaky_relu0p5(x: torch.Tensor) -> torch.Tensor:
    """Leaky ReLU with the negative slope = 0.5."""
    return torch.nn.functional.leaky_relu(x, negative_slope=0.5)


def leaky_relum0p5(x: torch.Tensor) -> torch.Tensor:
    """Leaky ReLU with the negative slope = -0.5."""
    return torch.nn.functional.leaky_relu(x, negative_slope=-0.5)


def inversed_leaky_relu0p5(x: torch.Tensor) -> torch.Tensor:
    """Inverse of leaky_relu0p5."""
    return torch.nn.functional.leaky_relu(x, negative_slope=2)


def truncated_atanh(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Inverse tanh with truncating values >=1 or <=-1."""
    x[x >= 1.0 - epsilon] = 1.0 - epsilon
    x[x <= -1.0 + epsilon] = -1.0 + epsilon
    return torch.atanh(x)


class SmoothLeakyReLU:
    """Smooth leaky ReLU"""

    def __init__(self, a: float = 0.75, b: float = 1 / 100):
        self.a = a
        self.b = b

    def __call__(self, x: torch.Tensor):
        # a x + (1 - a) sqrt(x**2 + b)
        return self.a * x + (1 - self.a) * torch.sqrt(x**2 + self.b)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.a * x
            - torch.sqrt(
                (self.a - 1) ** 2 * (2 * self.a * self.b - self.b + x**2)
            )
        ) / (2 * self.a - 1)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return self.a - ((self.a - 1) * x) / torch.sqrt(self.b + x**2)


def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """Gaussian Error Linear Units"""
    return torch.nn.functional.gelu(x, approximate=approximate)


_TensorLike = TypeVar("_TensorLike", torch.Tensor, IPhlowerTensor)


class ActivationSelector:
    _SMOOTH_LEAKY_RELU = SmoothLeakyReLU()

    _REGISTERED_ACTIVATIONS: dict[str, Callable[[_TensorLike], _TensorLike]] = {
        "gelu": gelu,
        "identity": identity,
        "inversed_leaky_relu0p5": inversed_leaky_relu0p5,
        "inversed_smooth_leaky_relu": _SMOOTH_LEAKY_RELU.inverse,
        "leaky_relu0p5": leaky_relu0p5,
        "leaky_relum0p5": leaky_relum0p5,
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "smooth_leaky_relu": _SMOOTH_LEAKY_RELU,
        "sqrt": torch.sqrt,
        "tanh": torch.tanh,
        "truncated_atanh": truncated_atanh,
    }

    @staticmethod
    def select(name: str) -> Callable[[_TensorLike], _TensorLike]:
        if name not in ActivationType.__members__:
            raise NotImplementedError(f"{name} is not implemented.")

        type_name = ActivationType[name]
        return ActivationSelector._REGISTERED_ACTIVATIONS[type_name.value]

    @staticmethod
    def select_inverse(
        name: str,
    ) -> Callable[[_TensorLike], _TensorLike]:
        if name not in ActivationType.__members__:
            raise NotImplementedError(f"{name} is not implemented.")

        type_name = ActivationType[name]
        return ActivationSelector._REGISTERED_ACTIVATIONS[
            ActivationSelector._inverse_activation_name(type_name).value
        ]

    @staticmethod
    def _inverse_activation_name(
        activation_name: ActivationType,
    ) -> ActivationType:
        _to_inverse: dict[ActivationType, ActivationType] = {
            ActivationType.identity: ActivationType.identity,
            ActivationType.leaky_relu0p5: ActivationType.inversed_leaky_relu0p5,
            ActivationType.smooth_leaky_relu: (
                ActivationType.inversed_smooth_leaky_relu
            ),
            ActivationType.tanh: ActivationType.truncated_atanh,
        }

        inverse_type = _to_inverse.get(activation_name)

        if inverse_type is None:
            raise PhlowerInvalidActivationError(
                f"Cannot inverse for {activation_name}"
            )
        return inverse_type

    @staticmethod
    def is_exists(name: str) -> bool:
        return name in ActivationSelector._REGISTERED_ACTIVATIONS
