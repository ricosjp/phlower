from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from phlower._base._dimension import PhysicalDimensions
from phlower._base.array._interface_wrapper import IPhlowerArray
from phlower._base.tensors import PhlowerTensor, phlower_tensor
from phlower.utils import get_logger
from phlower.utils.typing import SparseArrayType

logger = get_logger(__name__)


class SparseArrayWrapper(IPhlowerArray):
    def __init__(
        self, arr: SparseArrayType, dimensions: PhysicalDimensions | None = None
    ) -> None:
        self._sparse_data = arr
        self._dimensions = dimensions

    @property
    def is_sparse(self) -> bool:
        return True

    @property
    def is_time_series(self) -> bool:
        return False

    @property
    def is_voxel(self) -> bool:
        return False

    @property
    def row(self) -> np.ndarray:
        return self._sparse_data.row

    @property
    def col(self) -> np.ndarray:
        return self._sparse_data.col

    @property
    def data(self) -> np.ndarray:
        return self._sparse_data.data

    @property
    def shape(self) -> tuple[int]:
        return self._sparse_data.shape

    @property
    def size(self) -> int:
        return self._sparse_data.size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}, {self._sparse_data}"

    def apply(
        self,
        apply_function: Callable[[SparseArrayType], SparseArrayType],
        componentwise: bool,
        *,
        use_diagonal: bool = False,
        **kwards,
    ) -> SparseArrayType:
        if use_diagonal:
            raise ValueError(
                "Cannot set use_diagonal=True in self.apply function. "
                "use_diagonal is only allows in `reshape` method"
            )

        reshaped_array = self.reshape(
            componentwise=componentwise, use_diagonal=use_diagonal
        )

        result = apply_function(reshaped_array)
        return result.reshape(self.shape).tocoo()

    def reshape(
        self, componentwise: bool, *, use_diagonal: bool = False, **kwards
    ) -> SparseArrayType:
        if componentwise and use_diagonal:
            logger.warning(
                "component_wise and use_diagonal cannnot use"
                " at the same time. use_diagonal is ignored"
            )

        if componentwise:
            return self._sparse_data

        if use_diagonal:
            reshaped = self._sparse_data.diagonal()
            return reshaped

        reshaped = self._sparse_data.reshape((self.shape[0] * self.shape[1], 1))
        return reshaped

    def to_phlower_tensor(
        self,
        device: str | torch.device | None = None,
        non_blocking: bool = False,
        disable_dimensions: bool = False,
    ) -> PhlowerTensor:
        sparse_tensor = torch.sparse_coo_tensor(
            torch.stack(
                [
                    torch.LongTensor(self._sparse_data.row),
                    torch.LongTensor(self._sparse_data.col),
                ]
            ),
            torch.from_numpy(self._sparse_data.data),
            self._sparse_data.shape,
        )
        _tensor = phlower_tensor(
            tensor=sparse_tensor,
            dimension=None if disable_dimensions else self._dimensions,
            is_time_series=False,
            is_voxel=False,
        )
        _tensor = _tensor.coalesce()
        return _tensor.to(device=device, non_blocking=non_blocking)

    def to_numpy(self) -> SparseArrayType:
        return self._sparse_data

    def slice_along_time_axis(self, time_slice: slice):
        raise ValueError(
            "slice_along_time_axis is not allowed for sparse array."
        )
