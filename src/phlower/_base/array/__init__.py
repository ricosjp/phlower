from logging import getLogger
from typing import get_args

import numpy as np

from phlower._base import PhysicalDimensions
from phlower._base.array import dense, sparse
from phlower._base.array._interface_wrapper import IPhlowerArray
from phlower.utils.typing import ArrayDataType, DenseArrayType, SparseArrayType

_logger = getLogger(__name__)


def phlower_array(
    data: ArrayDataType | IPhlowerArray,
    is_time_series: bool = False,
    is_voxel: bool = False,
    dimensions: PhysicalDimensions | None = None,
) -> IPhlowerArray:
    if isinstance(data, IPhlowerArray):
        _logger.warning(
            "phlower_array function is called for PhlowerArray."
            "time_series and is_voxel in arguments are ignored."
        )
        return data

    if isinstance(data, DenseArrayType):
        return dense.NdArrayWrapper(
            data,
            is_time_series=is_time_series,
            is_voxel=is_voxel,
            dimensions=dimensions,
        )

    if isinstance(data, get_args(SparseArrayType)):
        if is_time_series or is_voxel:
            raise ValueError(
                "Sparse Array cannot have time series flag and voxel flag."
            )

        return sparse.SparseArrayWrapper(data, dimensions=dimensions)

    raise ValueError(f"Unsupported data type: {data.__class__}, {type(data)}")
