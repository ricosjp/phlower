import torch

from phlower.collections import phlower_tensor_collection
from phlower.collections.arrays import SequencedDictArray
from phlower.data._lumped_data import LumpedArrayData, LumpedTensorData


class PhlowerCollateFn:
    def __init__(
        self,
        device: str | torch.device,
        non_blocking: bool = False,
        disable_dimensions: bool = False,
    ) -> None:
        self._device = device
        self._non_blocking = non_blocking
        self._disable_dimensions = disable_dimensions

    def __call__(self, batch: list[LumpedArrayData]) -> LumpedTensorData:
        inputs = SequencedDictArray([v.x_data for v in batch])
        outputs = SequencedDictArray([v.y_data for v in batch])
        field_data = SequencedDictArray([v.field_data for v in batch])

        # concatenate and send
        inputs_tensors, inputs_batch_info = inputs.to_batched_tensor(
            device=self._device,
            non_blocking=self._non_blocking,
            disable_dimensions=self._disable_dimensions,
        )

        outputs_tensors, outputs_batch_info = outputs.to_batched_tensor(
            device=self._device,
            non_blocking=self._non_blocking,
            disable_dimensions=self._disable_dimensions,
        )
        field_tensors, field_batch_info = field_data.to_batched_tensor(
            device=self._device,
            non_blocking=self._non_blocking,
            disable_dimensions=self._disable_dimensions,
        )
        data_directories = [b.data_directory for b in batch]

        return LumpedTensorData(
            x_data=phlower_tensor_collection(inputs_tensors),
            y_data=phlower_tensor_collection(outputs_tensors),
            field_data=phlower_tensor_collection(field_tensors),
            data_directories=data_directories,
            x_batch_info=inputs_batch_info,
            y_batch_info=outputs_batch_info,
            field_batch_info=field_batch_info,
        )
