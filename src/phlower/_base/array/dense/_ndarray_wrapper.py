from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from phlower._base._dimension import PhysicalDimensions
from phlower._base.array._interface_wrapper import IPhlowerArray
from phlower.utils.typing import DenseArrayType


class NdArrayWrapper(IPhlowerArray):
    def __init__(
        self,
        data: np.ndarray,
        is_time_series: bool = False,
        is_voxel: bool = False,
        dimensions: PhysicalDimensions | None = None,
    ):
        self.data = data
        self._is_time_series = is_time_series
        self._is_voxel = is_voxel
        self._dimensions = dimensions

    @property
    def dimension(self) -> PhysicalDimensions | None:
        return self._dimensions

    @property
    def shape(self) -> tuple[int]:
        return self.data.shape

    @property
    def is_time_series(self) -> bool:
        return self._is_time_series

    @property
    def is_voxel(self) -> bool:
        return self._is_voxel

    @property
    def size(self) -> bool:
        return self.data.size

    @property
    def is_sparse(self) -> bool:
        return False

    def apply(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        componentwise: bool,
        *,
        skip_nan: bool = False,
        **kwards,
    ) -> IPhlowerArray:
        reshaped = self.reshape(
            componentwise=componentwise, skip_nan=skip_nan
        ).to_numpy()
        result = np.reshape(function(reshaped), self.shape)
        return NdArrayWrapper(
            result,
            is_time_series=self.is_time_series,
            is_voxel=self.is_voxel,
            dimensions=self._dimensions,
        )

    def reshape(
        self, componentwise: bool, *, skip_nan: bool = False, **kwards
    ) -> IPhlowerArray:
        result = self._reshape(
            componentwise=componentwise, skip_nan=skip_nan, **kwards
        )
        return NdArrayWrapper(
            result,
            is_time_series=self.is_time_series,
            is_voxel=self.is_voxel,
            dimensions=self._dimensions,
        )

    def _reshape(
        self, componentwise: bool, *, skip_nan: bool = False, **kwards
    ) -> np.ndarray:
        if componentwise:
            reshaped = np.reshape(
                self.data, (np.prod(self.shape[:-1]), self.shape[-1])
            )
        else:
            reshaped = np.reshape(self.data, (-1, 1))

        if not skip_nan:
            return reshaped

        if componentwise:
            isnan = np.isnan(np.prod(reshaped, axis=-1))
            reshaped_wo_nan = reshaped[~isnan]
            return reshaped_wo_nan
        else:
            reshaped_wo_nan = reshaped[~np.isnan(reshaped)][:, None]
            return reshaped_wo_nan

    def to_tensor(
        self,
        device: str | torch.device | None = None,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        return torch.from_numpy(self.data).to(
            device=device, non_blocking=non_blocking
        )

    def to_numpy(self) -> DenseArrayType:
        return self.data

    def slice_along_time_axis(self, time_slice: slice) -> DenseArrayType:
        if not self._is_time_series:
            raise ValueError(
                "slice_along_time_axis operation is not allowed "
                "for not time_series array."
            )
        return NdArrayWrapper(
            self.data[time_slice, ...],
            is_time_series=self._is_time_series,
            is_voxel=self._is_voxel,
            dimensions=self._dimensions,
        )
