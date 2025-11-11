from __future__ import annotations

import torch

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import ConcatenatorSetting


class Concatenator(IPhlowerCoreModule, torch.nn.Module):
    """
    This class is designed to concatenate multiple tensors
    along the last dimension and apply an activation function
    to the concatenated result.

    Parameters
    ----------
    nodes : list of int, optional
        List of feature dimension sizes (The last value of tensor shape).
        Defaults to None.
    activation : str, optional
        Name of the activation function to apply to the output.
        Defaults to "identity" (no activation).

    Examples
    --------
    >>> concatenator = Concatenator(activation="relu")
    >>> concatenator(data)

    """

    @classmethod
    def from_setting(cls, setting: ConcatenatorSetting) -> Concatenator:
        """Create Concatenator from setting object

        Args:
            setting (ConcatenatorSetting): setting object

        Returns:
            Self: Concatenator
        """
        return Concatenator(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Concatenator

        Returns:
            str: name
        """
        return "Concatenator"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self, activation: str, nodes: list[int] | None = None, **kwards
    ) -> None:
        super().__init__(**kwards)
        self._nodes = nodes
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
        return self._activation_func(torch.cat(tuple(data.values()), dim=-1))
