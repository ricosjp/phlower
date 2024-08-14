from collections.abc import Callable

import numpy as np
import torch

from phlower._base._dimension import PhysicalDimensions
from phlower._base.array._interface_wrapper import IPhlowerArray
from phlower._base.tensors import PhlowerTensor, phlower_tensor
from phlower.utils.typing import DenseArrayType


class NdArrayWrapper(IPhlowerArray):
    def __init__(self, data: np.ndarray, is_time_series: bool = False):
        self.data = data
        self._is_time_series = is_time_series

    @property
    def shape(self) -> tuple[int]:
        return self.data.shape

    @property
    def is_time_series(self) -> bool:
        return self._is_time_series

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
    ) -> np.ndarray:
        reshaped = self.reshape(componentwise=componentwise, skip_nan=skip_nan)
        result = function(reshaped)
        return np.reshape(result, self.shape)

    def reshape(
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

    def to_phlower_tensor(
        self,
        device: str | torch.device | None = None,
        non_blocking: bool = False,
        dimension: PhysicalDimensions | None = None,
        is_time_series: bool | None = None,
        is_voxel: bool = False,
    ) -> PhlowerTensor:
        if is_time_series is None:
            is_time_series = self.is_time_series
        _tensor = phlower_tensor(
            tensor=torch.from_numpy(self.data),
            dimension=dimension,
            is_time_series=is_time_series,
            is_voxel=is_voxel,
        )
        _tensor.to(device=device, non_blocking=non_blocking)
        return _tensor

    def to_numpy(self) -> DenseArrayType:
        return self.data
