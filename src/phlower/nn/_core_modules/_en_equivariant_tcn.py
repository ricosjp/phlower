from __future__ import annotations

import torch

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections import phlower_tensor_collection
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules._identity import Identity
from phlower.nn._core_modules._proportional import Proportional
from phlower.nn._core_modules._tcn import TCN
from phlower.nn._functionals import _functions
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import EnEquivariantTCNSetting


class EnEquivariantTCN(torch.nn.Module, IPhlowerCoreModule):
    """EnEquivariantTCN is a neural network module that performs
    an E(n)-equivariant operation on the time series input tensors.

    Parameters
    ----------
    nodes: list[int]
        List of feature dimension sizes (The last value of tensor shape).
    kernel_sizes: list[int]
        List of kernel sizes.
    dilations: list[int]
        List of dilations.
    activations: list[str]
        List of activation functions.
    bias: bool
        Whether to use bias.
    dropouts: list[float] | None (optional)
        List of dropout rates.
    create_linear_weight: bool
        Whether to create a linear weight. Default is True.

    Examples
    --------
    >>> en_equivariant_tcn = EnEquivariantTCN(
    ...     nodes=[10, 20, 30],
    ...     kernel_sizes=[3, 3, 3],
    ...     dilations=[1, 2, 4],
    ...     activations=["relu", "relu", "relu"],
    ...     bias=True,
    ...     dropouts=[0.1, 0.1, 0.1],
    ...     create_linear_weight=True)
    >>> en_equivariant_tcn(data)
    """

    @classmethod
    def from_setting(cls, setting: EnEquivariantTCNSetting) -> EnEquivariantTCN:
        """Generate model from setting object

        Args:
            setting (EnEquivariantMLPSetting): setting object

        Returns:
            EnEquivariantTCN: EnEquivariantTCN object
        """
        return cls(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "EnEquivariantTCN"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int],
        kernel_sizes: list[int],
        dilations: list[int],
        activations: list[str],
        bias: bool = True,
        dropouts: list[float] | None = None,
        create_linear_weight: bool = True,
    ) -> None:
        super().__init__()
        self._tcn = TCN(
            nodes=nodes,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            activations=activations,
            bias=bias,
            dropouts=dropouts,
        )

        if create_linear_weight:
            self._linear = Proportional(nodes=[nodes[0], nodes[-1]])
        else:
            if nodes[0] != nodes[-1]:
                raise ValueError(
                    "create_linear_weight must be True "
                    "when nodes[0] and nodes[-1] are different. "
                    f"{nodes}"
                )
            self._linear = Identity()

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

        x = data.unique_item()
        if not x.is_time_series:
            raise ValueError(
                "Only time series tensor is allowed in EnEquivariantTCN."
            )

        ts_norm = _functions.contraction(x)
        ts_norm_h = self._tcn.forward(
            phlower_tensor_collection({"_to_tcn": ts_norm})
        )

        h = self._linear.forward(data)

        return _functions.tensor_product(h, ts_norm_h)
