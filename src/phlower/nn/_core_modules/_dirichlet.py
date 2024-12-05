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
from phlower.settings._module_settings import DirichletSetting


class Dirichlet(IPhlowerCoreModule, torch.nn.Module):
    """Dirichlet"""

    @classmethod
    def from_setting(cls, setting: DirichletSetting) -> Self:
        """Create Dirichlet from setting object

        Args:
            setting (DirichletSetting): setting object

        Returns:
            Self: Dirichlet
        """
        return Dirichlet(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Dirichlet

        Returns:
            str: name
        """
        return "Dirichlet"

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
            supports (dict[str, PhlowerTensor], optional):
                Graph object. Defaults to None. Dirichlet will not use it.

        Returns:
            PhlowerTensor: Tensor object
        """
        dirichlet = tuple(data.values())[1]
        dirichlet_filter = torch.isnan(dirichlet)
        ans = torch.where(dirichlet_filter, tuple(data.values())[0], dirichlet)
        return self._activation_func(ans)
