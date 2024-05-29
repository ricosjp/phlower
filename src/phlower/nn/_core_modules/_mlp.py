from __future__ import annotations

import pydantic
import torch
from pydantic import Field
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._interface_module import (
    IPhlowerCoreModule
)
from phlower.nn._core_modules import _utils
from phlower.settings._module_settings import MLPSetting


class MLP(IPhlowerCoreModule, torch.nn.Module):
    """Multi Layer Perceptron."""

    @classmethod
    def from_setting(cls, setting: MLPSetting) -> Self:
        return MLP(**setting.__dict__)

    @classmethod
    def get_nn_name(cls):
        return "MLP"

    @classmethod
    def get_parameter_setting_cls(cls):
        return MLPSetting

    def __init__(
        self,
        nodes: list[int],
        activations: list[str] = None,
        dropouts: list[float] = None,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if activations is None:
            activations = []
        if dropouts is None:
            dropouts = []

        self._chains = _utils.ExtendedLinearList(
            nodes=nodes, activations=activations, dropouts=dropouts, bias=bias
        )
        self._nodes = nodes
        self._activations = activations

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor] = None,
    ) -> PhlowerTensor:
        h = data.unique_item()
        for i in range(len(self._chains)):
            h = self._chains.forward(h, index=i)
        return h
