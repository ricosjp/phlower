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
from phlower.settings._module_settings import MLPSetting


class MLP(torch.nn.Module, IPhlowerCoreModule):
    """Multi Layer Perceptron

    Parameters
    ----------
    nodes: list[int]
        List of feature dimension sizes (The last value of tensor shape).
    activations: list[str] | None (optional)
        List of activation functions to apply to the output.
        Defaults to None.
    dropouts: list[float] | None (optional)
        List of dropout rates to apply to the output.
        Defaults to None.
    bias: bool
        Whether to use bias.

    Examples
    --------
    >>> mlp = MLP(
    ...     nodes=[10, 20, 30],
    ...     activations=["relu", "relu", "relu"],
    ...     dropouts=[0.1, 0.1, 0.1],
    ...     bias=True
    ... )
    >>> mlp(data)
    """

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
        bias: bool = True,
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
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overloads torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """
        h = data.unique_item()
        h = self._chains.forward(h)
        return h
