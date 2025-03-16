from __future__ import annotations

import einops
import torch
from typing_extensions import Self

from phlower._base.tensors import IPhlowerTensor, phlower_tensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import RearrangeSetting


class Rearrange(IPhlowerCoreModule, torch.nn.Module):
    """Rearrange is a neural network module that performs a rearrange operation
    on the input tensor.

    Parameters
    ----------
    pattern: str
        Pattern of the rearrange operation.
    axes_lengths: dict[str, int]
        Axes lengths of the rearrange operation.
    save_as_time_series: bool
        Whether to save the output as a time series.
    save_as_voxel: bool
        Whether to save the output as a voxel.

    Examples
    --------
    >>> rearrange = Rearrange(
    ...     pattern="t n f-> n f t",
    ...     axes_lengths={"f": 10, "t": 10},
    ...     save_as_time_series=True,
    ...     save_as_voxel=True
    ... )
    >>> rearrange(data)
    """

    @classmethod
    def from_setting(cls, setting: RearrangeSetting) -> Self:
        """Create Rearrange from setting object

        Args:
            setting (RearrangeSetting): setting object

        Returns:
            Self: Rearrange
        """
        return Rearrange(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Rearrange

        Returns:
            str: name
        """
        return "Rearrange"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self,
        pattern: str,
        axes_lengths: dict[str, int],
        save_as_time_series: bool,
        save_as_voxel: bool,
        **kwards,
    ):
        super().__init__()
        self._pattern = pattern
        self._axes_lengths = axes_lengths
        self._save_as_time_series = save_as_time_series
        self._save_as_voxel = save_as_voxel

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
    ) -> IPhlowerTensor:
        """forward function which overloads torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """

        arg = data.unique_item()
        h = einops.rearrange(
            arg.to_tensor(), pattern=self._pattern, **self._axes_lengths
        )

        return phlower_tensor(
            h,
            dimension=arg.dimension,
            is_time_series=self._save_as_time_series,
            is_voxel=self._save_as_voxel,
        )
