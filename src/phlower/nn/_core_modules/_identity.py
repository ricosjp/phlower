from __future__ import annotations

import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import IdentitySetting


class Identity(IPhlowerCoreModule, torch.nn.Module):
    """Identity layer"""

    @classmethod
    def from_setting(cls, setting: IdentitySetting) -> Self:
        """Generate model from setting object

        Args:
            setting (EnEquivariantMLPSetting): setting object

        Returns:
            Self: Identity object
        """
        return cls(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "Identity"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(self, nodes: list[int] = None) -> None:
        super().__init__()
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
        supports: dict[str, PhlowerTensor] | None = None,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overloads torch.nn.Module

        Args:
            data (IPhlowerTensorCollections):
                data which receives from predecessors
            supports (dict[str, PhlowerTensor], optional):
                Graph object. Defaults to None.

        Returns:
            PhlowerTensor: Tensor object
        """
        return data.unique_item()
