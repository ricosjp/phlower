from __future__ import annotations

import warnings
from functools import reduce
from typing import Callable, Optional, Sequence

import numpy as np
import scipy.sparse as sp
import torch

from phlower._base._batch import SparseBatchInfo
from phlower._base.array._interface_wrapper import IPhlowerArray
from phlower._base.tensors import PhlowerTensor, phlower_tensor
from phlower.utils.typing import SparseArrayType


class SparseArrayWrapper(IPhlowerArray):
    def __init__(
        self, arr: SparseArrayType, batch_info: Optional[SparseBatchInfo] = None
    ) -> None:
        if batch_info is None:
            batch_info = SparseBatchInfo(
                sizes=[len(arr.data)], shapes=[arr.shape]
            )

        self._sparse_data = arr
        self._batch_info = batch_info

    @property
    def is_sparse(self) -> bool:
        return True

    @property
    def is_time_series(self):
        return False

    @property
    def row(self):
        return self._sparse_data.row

    @property
    def col(self):
        return self._sparse_data.col

    @property
    def data(self):
        return self._sparse_data.data

    @property
    def shape(self) -> tuple[int]:
        return self._sparse_data.shape

    @property
    def batch_info(self) -> SparseBatchInfo:
        return self._batch_info

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
                "use_diagonal is only allows in self.reshape"
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
            warnings.warn(
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

    def decompose(self) -> list[SparseArrayWrapper]:
        arrs = _sparse_decompose(self._sparse_data, self._batch_info)
        return [SparseArrayWrapper(arr) for arr in arrs]

    def to_phlower_tensor(
        self,
        device: str | torch.device | None,
        non_blocking: bool = False,
        dimension: dict[str, float] | None = None,
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
            dimension=dimension,
            sparse_batch_info=self._batch_info
        )
        _tensor.to(device=device, non_blocking=non_blocking)
        return _tensor

    def numpy(self) -> SparseArrayType:
        return self._sparse_data


def concatenate(arrays: Sequence[SparseArrayWrapper]):
    concat_arr = _sparse_concatenate(*arrays)

    concat_shapes = list(
        reduce(
            lambda x, y: x + y, [arr.batch_info.shapes for arr in arrays], []
        )
    )
    concat_sizes = list(
        reduce(lambda x, y: x + y, [arr.batch_info.sizes for arr in arrays], [])
    )

    info = SparseBatchInfo(sizes=concat_sizes, shapes=concat_shapes)
    return SparseArrayWrapper(concat_arr, info)


def decompose(array: SparseArrayWrapper):
    results = _sparse_decompose(array.numpy(), array.batch_info)
    return [SparseArrayWrapper(arr) for arr in results]


def _sparse_concatenate(*arrays: SparseArrayType):
    offsets = np.cumsum(
        np.array(
            [
                arrays[i - 1].shape if i != 0 else (0, 0)
                for i in range(len(arrays))
            ]
        ),
        axis=0,
    )

    shape = np.sum([arr.shape for arr in arrays], axis=0)

    rows = np.concatenate(
        [arr.row + offsets[i, 0] for i, arr in enumerate(arrays)],
        dtype=np.int32,
    )
    cols = np.concatenate(
        [arr.col + offsets[i, 1] for i, arr in enumerate(arrays)],
        dtype=np.int32,
    )
    data = np.concatenate([arr.data for arr in arrays], dtype=np.float32)

    sparse_arr = sp.coo_matrix((data, (rows, cols)), shape=shape)
    return sparse_arr


def _sparse_decompose(array: SparseArrayType, batch_info: SparseBatchInfo):
    sizes = np.cumsum(batch_info.sizes)
    offsets = np.cumsum(
        np.array([[0, 0]] + batch_info.shapes[:-1], dtype=np.int32), axis=0
    )

    rows = np.split(array.row, sizes)
    cols = np.split(array.col, sizes)
    data = np.split(array.data, sizes)

    results = [
        sp.coo_matrix(
            (data[i], (rows[i] - offsets[i][0], cols[i] - offsets[i][1])),
            shape=batch_info.shapes[i],
        )
        for i in range(len(batch_info))
    ]
    return results
