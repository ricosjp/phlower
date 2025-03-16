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
from phlower.settings._module_settings import EinsumSetting


class Einsum(IPhlowerCoreModule, torch.nn.Module):
    """Einsum is a neural network module that performs a einsum operation
    on the input tensors.

    Parameters
    ----------
    activation: str
        Name of the activation function to apply to the output.
    equation: str
        Equation of the einsum operation.

    Examples
    --------
    >>> einsum = Einsum(activation="relu", equation="ij,jk->ik")
    >>> einsum(data)
    """

    @classmethod
    def from_setting(cls, setting: EinsumSetting) -> Einsum:
        """Create Einsum from setting object

        Args:
            setting (EinsumSetting): setting object

        Returns:
            Self: Einsum
        """
        return Einsum(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Einsum

        Returns:
            str: name
        """
        return "Einsum"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(self, activation: str, equation: str, **kwards) -> None:
        super().__init__()
        self._activation_name = activation
        self._activation_func = _utils.ActivationSelector.select(activation)
        self._equation = equation

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

        return self._activation_func(
            _functions.einsum(self._equation, *data.values())
        )
