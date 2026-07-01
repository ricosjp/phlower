from __future__ import annotations

from typing import Self

import torch
from phlower_tensor import ISimulationField, PhlowerTensor
from phlower_tensor.collections import IPhlowerTensorCollections

from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import FixedNormSetting


class FixedNorm(IPhlowerCoreModule, torch.nn.Module):
    """FixedNorm is a neural network module that applies
    normalization on the input tensor with fixed parameters.

    Parameters
    ----------
    nodes: list[int]
        List of feature dimension sizes (The last value of tensor shape).
    mean_name: str
        Name of the mean parameter for normalization.
    std_name: str
        Name of the standard deviation parameter for normalization.

    Examples
    --------
    >>> fixed_norm = FixedNorm(mean_name="mean", std_name="std")
    >>> fixed_norm(data)
    """

    @classmethod
    def from_setting(cls, setting: FixedNormSetting) -> Self:
        """Generate FixedNorm from setting object

        Args:
            setting (FixedNormSetting): setting object

        Returns:
            Self: FixedNorm object
        """
        return FixedNorm(**setting.model_dump())

    @classmethod
    def get_nn_name(cls) -> str:
        """Return neural network name

        Returns:
            str: name
        """
        return "FixedNorm"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        nodes: list[int] | None = None,
        mean_name: str = "mean",
        std_name: str = "std",
    ) -> None:
        super().__init__()
        self._nodes = nodes
        self._mean_name = mean_name
        self._std_name = std_name

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
        h = data.unique_item()

        if field_data is None:
            raise ValueError(
                "field_data must be provided for FixedNorm module."
            )

        if (
            self._mean_name not in field_data
            or self._std_name not in field_data
        ):
            raise KeyError(
                f"Mean or std not found in field_data. "
                f"Expected keys: {self._mean_name}, {self._std_name}. "
                f"Available keys: {list(field_data.keys())}"
            )

        _mean = field_data[self._mean_name]
        _std = field_data[self._std_name]

        _orig_shape = h.shape
        h = (h - _mean) / _std

        return h
