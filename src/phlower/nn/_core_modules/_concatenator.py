from __future__ import annotations

import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._interface_module import IPhlowerCoreModule
from phlower.settings._module_settings import ConcatenatorSetting

# from phlower.settings


class Concatenator(IPhlowerCoreModule, torch.nn.Module):
    """Concatenator"""

    @classmethod
    def from_setting(cls, setting: ConcatenatorSetting) -> Self:
        """Create Concatenator from setting object

        Args:
            setting (ConcatenatorSetting): setting object

        Returns:
            Self: Concatenator
        """
        return Concatenator(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Concatenator

        Returns:
            str: name
        """
        return "Concatenator"

    def __init__(self, activation: str, nodes: list[int] = None):
        super().__init__()
        self._nodes = nodes
        self._activation_name = activation
        self._activation_func = _utils.ActivationSelector.select(activation)

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor] | None = None,
    ) -> PhlowerTensor:
        """forward function which overloads torch.nn.Module

        Args:
            data (IPhlowerTensorCollections):
                data which receives from predecessors
            supports (dict[str, PhlowerTensor], optional):
                Graph object. Defaults to None. Concatenator will not use it.

        Returns:
            PhlowerTensor: Tensor object
        """
        return self._activation_func(torch.cat(tuple(data.values()), dim=-1))
