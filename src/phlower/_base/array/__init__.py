# flake8: noqa
from typing import get_args

from phlower._base.array._interface_wrapper import IPhlowerArray
from phlower._base.array._ndarray_wrapper import NdArrayWrapper
from phlower._base.array._sparse_array_wrapper import SparseArrayWrapper
from phlower.utils.typing import ArrayDataType, DenseArrayType, SparseArrayType


def phlower_arrray(data: ArrayDataType) -> IPhlowerArray:
    if isinstance(data, DenseArrayType):
        return NdArrayWrapper(data)

    if isinstance(data, get_args(SparseArrayType)):
        return SparseArrayWrapper(data)

    raise ValueError(f"Unsupported data type: {data.__class__}")
