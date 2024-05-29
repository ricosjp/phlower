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
from phlower.settings._module_settings import ConcatenatorSetting

# from phlower.settings

class Concatenator(IPhlowerCoreModule, torch.nn.Module):
    @classmethod
    def from_setting(cls, setting: ConcatenatorSetting) -> Self:
        return Concatenator(**setting.__dict__)

    @classmethod
    def get_nn_name(cls):
        return "Concatenator"

    @classmethod
    def get_parameter_setting_cls(cls):
        return ConcatenatorSetting

    def __init__(self, activation: str, nodes: list[int] = None):
        self._nodes = nodes
        self._activation_name = activation
        self._activation_func = _utils.ActivationSelector.select(activation)

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor] = None,
    ) -> PhlowerTensor:
        return self._activation_func(torch.cat(data.values(), dim=-1))
