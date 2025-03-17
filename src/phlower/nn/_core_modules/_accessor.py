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
from phlower.settings._module_settings import AccessorSetting


class Accessor(IPhlowerCoreModule, torch.nn.Module):
    """Accessor is a neural network module that provides access to specific
    tensor data with optional activation.

    This module is designed to access and process specific elements
    from input tensor collections. It can extract data at a specified index
    and apply an activation function to the result.

    Parameters
    ----------
    nodes: list[int] | None (optional)
        List of feature dimension sizes (The last value of tensor shape).
        Defaults to None.
    activation: str (optional)
        Name of the activation function to apply to the output.
        Defaults to "identity" (no activation).
    index: int (optional)
        The index to access from the input tensor. Defaults to 0.

    Examples
    --------
    >>> accessor = Accessor(activation="relu", index=2)
    >>> output = accessor(input_data)  # Applies activation to data[2]

    """

    @classmethod
    def from_setting(cls, setting: AccessorSetting) -> Self:
        """Create Accessor from setting object

        Args:
            setting (AccessorSetting): setting object

        Returns:
            Self: Accessor
        """
        return cls(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Accessor

        Returns:
            str: name
        """
        return "Accessor"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int] | None = None,
        activation: str = "identity",
        index: int = 0,
    ) -> None:
        super().__init__()
        self._nodes = nodes
        self._activation_name = activation
        self._activation_func = _utils.ActivationSelector.select(activation)
        self._index = index

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
        return self._activation_func(data.unique_item()[self._index])
