import pathlib

import numpy as np

from phlower.base.tensors import PhlowerTensor
from phlower.utils.typing import ArrayDataType, SparseArrayType


class BunchedData:
    def __init__(
        self,
        x_data: dict[str, ArrayDataType],
        y_data: dict[str, ArrayDataType],
        sparse_supports: dict[str, SparseArrayType],
        data_directory: pathlib.Path,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.sparse_supports = sparse_supports
        self.data_directory = data_directory

    # TODO Check data type is same in each item


class BunchedTensorData:
    def __init__(
        self,
        x_data: dict[str, PhlowerTensor],
        y_data: dict[str, PhlowerTensor],
        sparse_supports: dict[str, PhlowerTensor],
        data_directory: pathlib.Path,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.sparse_supports = sparse_supports
        self.data_directory = data_directory
