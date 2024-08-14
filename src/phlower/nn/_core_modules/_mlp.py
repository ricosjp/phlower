from __future__ import annotations

import torch
from typing_extensions import Self

from phlower import ISimulationField
from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import MLPSetting


class MLP(IPhlowerCoreModule, torch.nn.Module):
    """Multi Layer Perceptron"""

    @classmethod
    def from_setting(cls, setting: MLPSetting) -> Self:
        """Generate MLP from setting object

        Args:
            setting (MLPSetting): setting object

        Returns:
            Self: MLP object
        """
        return MLP(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "MLP"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int],
        activations: list[str] | None = None,
        dropouts: list[float] | None = None,
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
        h = data.unique_item()
        for i in range(len(self._chains)):
            h = self._chains.forward(h, index=i)
        return h
