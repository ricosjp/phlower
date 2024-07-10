import pathlib

from phlower import IPhlowerArray
from phlower._base import GraphBatchInfo
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


class LumpedTensorData:
    def __init__(
        self,
        x_data: IPhlowerTensorCollections,
        sparse_supports: IPhlowerTensorCollections,
        data_directories: list[pathlib.Path] | None = None,
        y_data: IPhlowerTensorCollections | None = None,
        x_batch_info: dict[str, GraphBatchInfo] | None = None,
        y_batch_info: dict[str, GraphBatchInfo] | None = None,
        supports_batch_info: dict[str, GraphBatchInfo] | None = None,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.sparse_supports = sparse_supports
        self.data_directories = data_directories

        self.x_batch_info = x_batch_info
        self.y_batch_info = y_batch_info
        self.supports_batch_info = supports_batch_info
