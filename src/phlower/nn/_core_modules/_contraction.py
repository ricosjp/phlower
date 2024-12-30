from __future__ import annotations

import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _functions, _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import ContractionSetting


class Contraction(IPhlowerCoreModule, torch.nn.Module):
    """Contraction"""

    @classmethod
    def from_setting(cls, setting: ContractionSetting) -> Self:
        """Create Contraction from setting object

        Args:
            setting (ContractionSetting): setting object

        Returns:
            Self: Contraction
        """
        return Contraction(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Contraction

        Returns:
            str: name
        """
        return "Contraction"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(self, activation: str, nodes: list[int] = None):
        super().__init__()
        self._nodes = nodes
        self._activation_name = activation
        self._activation_func = _utils.ActivationSelector.select(activation)

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
        return self._activation_func(_functions.contraction(*data.values()))
