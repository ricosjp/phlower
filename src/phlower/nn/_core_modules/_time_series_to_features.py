from __future__ import annotations

import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._functionals import _functions
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import TimeSeriesToFeaturesSetting


class TimeSeriesToFeatures(IPhlowerCoreModule, torch.nn.Module):
    """
    TimeSeriesToFeatures is a neural network module that converts a time series
    tensor into a non time series tensor by summing the time series tensor
    along the time dimension.

    Parameters
    ----------
    nodes: list[int]
        List of feature dimension sizes (The last value of tensor shape).
    activation: str
        Activation function to apply to the output.

    Examples
    --------
    >>> time_series_to_features = TimeSeriesToFeatures(
    ...     nodes=[10, 20, 30],
    ...     activation="relu",
    ... )
    >>> time_series_to_features(data)
    """

    @classmethod
    def from_setting(cls, setting: TimeSeriesToFeaturesSetting) -> Self:
        """Create TimeSeriesToFeatures from setting object

        Args:
            setting (TimeSeriesToFeaturesSetting): setting object

        Returns:
            Self: TimeSeriesToFeatures
        """
        return cls(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of TimeSeriesToFeatures

        Returns:
            str: name
        """
        return "TimeSeriesToFeatures"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int] = None,
        activation: str = "identity",
    ) -> None:
        super().__init__()
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
                data which receives from predecessor
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """

        return self._activation_func(
            _functions.time_series_to_features(data.unique_item())
        )
