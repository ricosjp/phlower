from __future__ import annotations

from typing import Any, NamedTuple, TypeVar

import numpy as np
import torch

from phlower._base.tensors import PhlowerTensor
from phlower.nn._functionals._activations import ActivationSelector
from phlower.utils.enums import ActivationType

T = TypeVar("T")


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

    def get_activation_names(self) -> list[str]:
        return self._activations

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


class ExtendedConv1DList(torch.nn.Module):
    def __init__(
        self,
        nodes: list[int],
        kernel_sizes: list[int],
        dilations: list[int],
        activations: list[str],
        bias: bool,
        dropouts: list[float] | None = None,
    ) -> None:
        super().__init__()

        assert len(nodes) >= 2
        self._nodes = nodes
        self._activations = self._validate_args(activations, "identity")
        self._kernel_sizes = self._validate_args(kernel_sizes, None)
        self._dilations = self._validate_args(dilations, 1)
        self._dropouts = self._validate_args((dropouts or []), 0)

        self._n_chains = len(self._nodes)
        self._bias = bias
        self._convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(n1, n2, k, bias=bias, dilation=d)
                for n1, n2, k, d in zip(
                    self._nodes[:-1],
                    self._nodes[1:],
                    self._kernel_sizes,
                    self._dilations,
                    strict=False,
                )
            ]
        )
        self._activators = [
            ActivationSelector.select(name) for name in self._activations
        ]

    def __len__(self) -> int:
        return len(self._convs)

    def __getitem__(self, idx: int) -> torch.nn.Linear:
        return self._convs[idx]

    def get_padding_lengths(self) -> np.ndarray:
        return (np.array(self._kernel_sizes) - 1) * np.array(self._dilations)

    def forward(self, x: PhlowerTensor) -> PhlowerTensor:
        lengths = self.get_padding_lengths()
        h = x
        for i in range(self._n_chains - 1):
            h = torch.nn.functional.pad(h, (lengths[i], 0), mode="replicate")
            h = self._convs[i].forward(h)
            h = torch.nn.functional.dropout(
                h, p=self._dropouts[i], training=self.training
            )
            h = self._activators[i](h)
        return h

    def _validate_args(self, values: list[T], default: T | None) -> list[T]:
        if len(values) == 0:
            values = [default for _ in range(len(self._nodes) - 1)]
        assert len(self._nodes) == len(values) + 1, f"{values} is too many."
        return values
