# flake8: noqa
from typing import get_args

from phlower.utils.typing import ArrayDataType, DenseArrayType, SparseArrayType

from .interface_wrapper import IPhlowerArray
from .ndarray_wrapper import NdArrayWrapper
from .sparse_array_wrapper import SparseArrayWrapper


def phlower_arrray(data: ArrayDataType) -> IPhlowerArray:
    if isinstance(data, DenseArrayType):
        return NdArrayWrapper(data)

    if isinstance(data, get_args(SparseArrayType)):
        return SparseArrayWrapper(data)

    raise ValueError(f"Unsupported data type: {data.__class__}")
