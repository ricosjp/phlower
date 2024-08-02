# flake8: noqa
from typing import get_args

from phlower._base.array import dense, sparse
from phlower._base.array._interface_wrapper import IPhlowerArray
from phlower.utils.typing import ArrayDataType, DenseArrayType, SparseArrayType


def phlower_array(data: ArrayDataType | IPhlowerArray) -> IPhlowerArray:
    if isinstance(data, IPhlowerArray):
        return data

    if isinstance(data, DenseArrayType):
        return dense.NdArrayWrapper(data)

    if isinstance(data, get_args(SparseArrayType)):
        return sparse.SparseArrayWrapper(data)

    raise ValueError(f"Unsupported data type: {data.__class__}, {type(data)}")
