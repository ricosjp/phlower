from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from phlower._base.tensors import PhlowerTensor
from phlower.nn._core_modules import _functions
from phlower.utils.enums import ActivationType
from phlower.utils.exceptions import PhlowerInvalidActivationError


class MLPConfiguration(NamedTuple):
    nodes: list[int]
    activations: list[str]
    dropouts: list[float]
    bias: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "activations": self.activations,
            "dropouts": self.dropouts,
            "bias": self.bias,
        }


class ExtendedLinearList(torch.nn.Module):
    @classmethod
    def from_config(cls, config: MLPConfiguration) -> ExtendedLinearList:
        return ExtendedLinearList(**config.to_dict())

    def __init__(
        self,
        nodes: list[int],
        activations: list[str],
        bias: bool,
        dropouts: list[float] | None = None,
    ) -> None:
        super().__init__()

        self._nodes = nodes
        self._activations = activations
        if dropouts is None:
            dropouts = []
        self._dropouts = dropouts
        self._validate_args()

        self._n_chains = len(self._nodes)
        self._is_bias = bias
        self._linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(n1, n2, bias=bias)
                for n1, n2 in zip(
                    self._nodes[:-1], self._nodes[1:], strict=False
                )
            ]
        )
        self._activators = [
            ActivationSelector.select(name) for name in self._activations
        ]

        # for m in self._linears:
        #     torch.nn.init.xavier_uniform_(m.weight.data)

    def __len__(self) -> int:
        return len(self._linears)

    def __getitem__(self, idx: int) -> torch.nn.Linear:
        return self._linears[idx]

    def has_nonlinearity(self) -> bool:
        if self._is_bias:
            return True

        if self.has_nonlinear_activations():
            return True

        if self.has_dropout():
            return True

        return False

    def has_bias(self) -> bool:
        return self._is_bias

    def has_nonlinear_activations(self) -> bool:
        n_nonlinear = sum(
            1 for v in self._activations if v != ActivationType.identity
        )
        return n_nonlinear > 0

    def has_dropout(self) -> bool:
        if len(self._dropouts) == 0:
            return False

        return sum(self._dropouts) > 0

    def forward_part(self, x: PhlowerTensor, *, index: int) -> PhlowerTensor:
        assert index < self._n_chains

        h = self._linears[index](x)
        h = torch.nn.functional.dropout(
            h, p=self._dropouts[index], training=self.training
        )
        h = self._activators[index](h)
        return h

    def forward(self, x: PhlowerTensor) -> PhlowerTensor:
        h = x
        for i in range(self._n_chains - 1):
            h = self.forward_part(h, index=i)
        return h

    def _validate_args(self) -> None:
        assert len(self._nodes) >= 2

        if len(self._activations) == 0:
            self._activations = [
                "identity" for _ in range(len(self._nodes) - 1)
            ]
        assert len(self._nodes) == len(self._activations) + 1

        if len(self._dropouts) == 0:
            self._dropouts = [0 for _ in range(len(self._nodes) - 1)]
        for v in self._dropouts:
            assert v < 1.0
        assert len(self._nodes) == len(self._dropouts) + 1


class ActivationSelector:
    _SMOOTH_LEAKY_RELU = _functions.SmoothLeakyReLU()

    _REGISTERED_ACTIVATIONS = {
        "identity": _functions.identity,
        "inversed_leaky_relu0p5": _functions.inversed_leaky_relu0p5,
        "inversed_smooth_leaky_relu": _SMOOTH_LEAKY_RELU.inverse,
        "leaky_relu0p5": _functions.leaky_relu0p5,
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "smooth_leaky_relu": _SMOOTH_LEAKY_RELU,
        "sqrt": torch.sqrt,
        "tanh": torch.tanh,
        "truncated_atanh": _functions.truncated_atanh,
    }

    @staticmethod
    def select(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        type_name = ActivationType[name]
        return ActivationSelector._REGISTERED_ACTIVATIONS[type_name.value]

    @staticmethod
    def select_inverse(
        name: str,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
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
