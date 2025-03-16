from __future__ import annotations

import torch

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._core_modules._identity import Identity
from phlower.nn._core_modules._proportional import Proportional
from phlower.nn._functionals import _functions
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import EnEquivariantMLPSetting


class EnEquivariantMLP(IPhlowerCoreModule, torch.nn.Module):
    """EnEquivariantMLP is a neural network module that performs
    an E(n)-equivariant operation on the input tensor.

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
        Whether to use bias in the output.
    create_linear_weight: bool
        Whether to create a linear weight. Default is False.
    norm_function_name: str
        Name of the normalization function to apply to the output.

    Examples
    --------
    >>> en_equivariant_mlp = EnEquivariantMLP(
    ...     nodes=[10, 20, 30],
    ...     activations=["relu", "relu", "relu"],
    ...     dropouts=[0.1, 0.1, 0.1],
    ...     bias=True,
    ...     create_linear_weight=True,
    ...     norm_function_name="identity")
    >>> en_equivariant_mlp(data)
    """

    @classmethod
    def from_setting(cls, setting: EnEquivariantMLPSetting) -> EnEquivariantMLP:
        """Generate model from setting object

        Args:
            setting (EnEquivariantMLPSetting): setting object

        Returns:
            Self: EnEquivariantMLP object
        """
        return EnEquivariantMLP(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "EnEquivariantMLP"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int],
        activations: list[str] | None = None,
        dropouts: list[float] | None = None,
        bias: bool = True,
        create_linear_weight: bool = False,
        norm_function_name: str = "identity",
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
        self._create_linear_weight = create_linear_weight
        self._norm_function_name = norm_function_name

        self._linear_weight = self._init_linear_weight()
        self._norm_function = _utils.ActivationSelector.select(
            self._norm_function_name
        )

    def _init_linear_weight(self) -> Identity | Proportional:
        if not self._create_linear_weight:
            if self._nodes[0] != self._nodes[-1]:
                raise ValueError(
                    "First and last nodes are different. "
                    "Set create_linear_weight True."
                )
            return Identity()

        return Proportional([self._nodes[0], self._nodes[-1]])

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
        linear_x = self._linear_weight(data)
        h = self._chains(self._norm_function(_functions.contraction(x)))
        return _functions.tensor_product(linear_x, h)
