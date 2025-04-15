from logging import getLogger
from typing import Any, get_args

import numpy as np

from phlower._base import PhysicalDimensions
from phlower._base.array import dense, sparse
from phlower._base.array._interface_wrapper import IPhlowerArray
from phlower.utils.typing import ArrayDataType, DenseArrayType, SparseArrayType

_logger = getLogger(__name__)


def phlower_array(
    data: ArrayDataType | IPhlowerArray,
    is_time_series: bool | None = None,
    is_voxel: bool | None = None,
    dimensions: PhysicalDimensions | None = None,
    dtype: np.dtype = np.float32,
) -> IPhlowerArray:
    if isinstance(data, IPhlowerArray):
        if not _is_all_none(is_time_series, is_voxel, dimensions):
            _logger.warning(
                "phlower_array function is called for PhlowerArray."
                "dimensions, time_series and is_voxel in arguments "
                "are ignored."
            )
        return data

    is_time_series = bool(is_time_series)
    is_voxel = bool(is_voxel)

    if isinstance(data, DenseArrayType):
        return dense.NdArrayWrapper(
            data.astype(dtype),
            is_time_series=is_time_series,
            is_voxel=is_voxel,
            dimensions=dimensions,
        )

    if isinstance(data, get_args(SparseArrayType)):
        if is_time_series or is_voxel:
            raise ValueError(
                "Sparse Array cannot have time series flag and voxel flag."
            )

        return sparse.SparseArrayWrapper(
            data.astype(dtype), dimensions=dimensions
        )

    raise ValueError(f"Unsupported data type: {data.__class__}, {type(data)}")


def _is_all_none(*args: Any) -> bool:
    return all(arg is None for arg in args)
