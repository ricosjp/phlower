from __future__ import annotations

import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _functions, _utils
from phlower.nn._core_modules._identity import Identity
from phlower.nn._core_modules._proportional import Proportional
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import EnEquivariantMLPSetting


class EnEquivariantMLP(IPhlowerCoreModule, torch.nn.Module):
    """E(n)-equivariant Multi Layer Perceptron"""

    @classmethod
    def from_setting(cls, setting: EnEquivariantMLPSetting) -> Self:
        """Generate model from setting object

        Args:
            setting (EnEquivariantMLPSetting): setting object

        Returns:
            Self: EnEquivariantMLP object
        """
        return cls(**setting.__dict__)

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
        bias: bool = False,
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

    def _init_linear_weight(self):
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
        x = data.unique_item()
        linear_x = self._linear_weight(data)
        h = self._chains(self._norm_function(_functions.contraction(x)))
        return _functions.tensor_product(linear_x, h)
