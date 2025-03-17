from __future__ import annotations

import torch

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._functionals import _functions
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import GCNSetting


class GCN(IPhlowerCoreModule, torch.nn.Module):
    """GCN is a neural network module that performs a graph convolution
    operation on the input tensor.

    Parameters
    ----------
    nodes: list[int]
        List of feature dimension sizes (The last value of tensor shape).
    support_name: str
        Name of the support tensor.
    activations: list[str] | None (optional)
        List of activation functions to apply to the output.
        Defaults to None.
    dropouts: list[float] | None (optional)
        List of dropout rates to apply to the output.
        Defaults to None.
    repeat: int
        Number of times to repeat the operation.
    factor: float
        Factor to multiply the support tensor by.
    bias: bool
        Whether to use bias.

    Examples
    --------
    >>> gcn = GCN(
    ...     nodes=[10, 20, 30],
    ...     support_name="support",
    ...     activations=["relu", "relu", "relu"],
    ...     dropouts=[0.1, 0.1, 0.1],
    ...     repeat=1,
    ...     factor=1.0,
    ...     bias=True
    ...)
    >>> gcn(data)
    """

    @classmethod
    def from_setting(cls, setting: GCNSetting) -> GCN:
        """Create GCN from GCNSetting instance

        Args:
            setting (GCNSetting): setting object for GCN

        Returns:
            GCN: GCN object
        """
        return GCN(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "GCN"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int],
        support_name: str,
        activations: list[str] | None = None,
        dropouts: list[float] | None = None,
        repeat: int = 1,
        factor: float = 1.0,
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
        self._support_name = support_name
        self._repeat = repeat
        self._factor = factor

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overload torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor:
                Tensor object
        """
        support = field_data[self._support_name]
        h = data.unique_item()
        for i in range(len(self._chains)):
            h = self._propagate(h, support)
            h = self._chains.forward_part(h, index=i)
        return h

    def _propagate(
        self, x: PhlowerTensor, support: PhlowerTensor
    ) -> PhlowerTensor:
        h = _functions.spmm(support * self._factor, x, repeat=self._repeat)
        return h
