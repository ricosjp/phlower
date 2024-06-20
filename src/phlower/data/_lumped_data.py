import pathlib

from phlower import IPhlowerArray
from phlower.collections.tensors import IPhlowerTensorCollections


class LumpedArrayData:
    def __init__(
        self,
        x_data: dict[str, IPhlowerArray],
        y_data: dict[str, IPhlowerArray],
        sparse_supports: dict[str, IPhlowerArray],
        data_directory: pathlib.Path,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.sparse_supports = sparse_supports
        self.data_directory = data_directory

    # TODO Check data type is same in each item


class LumpedTensorData:
    def __init__(
        self,
        x_data: IPhlowerTensorCollections,
        y_data: IPhlowerTensorCollections,
        sparse_supports: IPhlowerTensorCollections,
        data_directory: pathlib.Path,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.sparse_supports = sparse_supports
        self.data_directory = data_directory
