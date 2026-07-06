from __future__ import annotations

import torch
from phlower_tensor import ISimulationField, PhlowerTensor
from phlower_tensor.collections import IPhlowerTensorCollections

from phlower.nn._core_modules import _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import ActivationSetting


class Activation(IPhlowerCoreModule, torch.nn.Module):
    """Activation is a module to apply an activation function.

    Parameters
    ----------
    activation: str
        Name of the activation function to apply to the output.

    Examples
    --------
    >>> activation = Activation(activation="relu")
    >>> activation(data)
    """

    @classmethod
    def from_setting(cls, setting: ActivationSetting) -> Activation:
        """Create Einsum from setting object

        Args:
            setting (EinsumSetting): setting object

        Returns:
            Self: Einsum
        """
        return Activation(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Activation

        Returns:
            str: name
        """
        return "Activation"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(self, activation: str, **kwards) -> None:
        super().__init__()
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
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """
        h = data.unique_item()
        return self._activation_func(h)
