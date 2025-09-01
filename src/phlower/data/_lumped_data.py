from __future__ import annotations

import torch

from phlower import IPhlowerArray
from phlower._base import GraphBatchInfo
from phlower._fields import SimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.io import PhlowerDirectory


class LumpedArrayData:
    def __init__(
        self,
        x_data: dict[str, IPhlowerArray],
        y_data: dict[str, IPhlowerArray],
        field_data: dict[str, IPhlowerArray],
        data_directory: PhlowerDirectory | None = None,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.field_data = field_data
        self.data_directory = data_directory


class LumpedTensorData:
    def __init__(
        self,
        x_data: IPhlowerTensorCollections,
        field_data: IPhlowerTensorCollections | SimulationField,
        data_directories: list[PhlowerDirectory] | None = None,
        y_data: IPhlowerTensorCollections | None = None,
        x_batch_info: dict[str, GraphBatchInfo] | None = None,
        y_batch_info: dict[str, GraphBatchInfo] | None = None,
        field_batch_info: dict[str, GraphBatchInfo] | None = None,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.data_directories = data_directories

        self.x_batch_info = x_batch_info
        self.y_batch_info = y_batch_info

        if isinstance(field_data, SimulationField):
            self.field_data = field_data
        else:
            self.field_data = SimulationField(
                field_tensors=field_data, batch_info=field_batch_info
            )

    def to(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> LumpedTensorData:
        return LumpedTensorData(
            x_data=self.x_data.to(device, non_blocking=non_blocking),
            y_data=(
                self.y_data.to(device, non_blocking=non_blocking)
                if self.y_data is not None
                else None
            ),
            field_data=self.field_data.to(device, non_blocking=non_blocking),
            data_directories=self.data_directories,
            x_batch_info=self.x_batch_info,
            y_batch_info=self.y_batch_info,
            field_batch_info=self.field_data._batch_info,
        )

    @property
    def n_data(self) -> int:
        if self.data_directories is None:
            return 1

        return len(self.data_directories)
