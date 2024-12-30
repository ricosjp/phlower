from __future__ import annotations

import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import ProportionalSetting


class Proportional(IPhlowerCoreModule, torch.nn.Module):
    """Proportional, i.e., strictly linear, layer"""

    @classmethod
    def from_setting(cls, setting: ProportionalSetting) -> Self:
        """Generate model from setting object

        Args:
            setting (ProportionalSetting): setting object

        Returns:
            Self: Proportional object
        """
        return cls(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "Proportional"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int],
        dropouts: list[float] | None = None,
    ) -> None:
        super().__init__()

        if dropouts is None:
            dropouts = []

        self._chains = _utils.ExtendedLinearList(
            nodes=nodes, activations=["identity"], dropouts=[], bias=False
        )
        self._nodes = nodes

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overloads torch.nn.Module

        Args:
            data (IPhlowerTensorCollections):
                data which receives from predecessors
            field_data (ISimulationField):
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """
        h = data.unique_item()
        return self._chains(h)
