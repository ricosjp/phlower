import torch
from phlower_tensor import GraphBatchInfo, IPhlowerArray, ISimulationField
from phlower_tensor.collections import (
    SequencedDictArray,
    phlower_tensor_collection,
)
from phlower_tensor.functionals import to_batch

from phlower.data._lumped_data import LumpedArrayData, LumpedTensorData
from phlower.utils._extended_simulation_field import (
    create_simulation_field,
)

from ._utils import BatchModeHolder


class PhlowerCollateFn:
    def __init__(
        self,
        device: str | torch.device,
        non_blocking: bool = False,
        disable_dimensions: bool = False,
        batch_mode_holder: BatchModeHolder | None = None,
    ) -> None:
        self._device = device
        self._non_blocking = non_blocking
        self._disable_dimensions = disable_dimensions
        self._batch_mode_holder = batch_mode_holder or BatchModeHolder(
            inputs_batch_mode={},
            labels_batch_mode={},
            field_batch_mode={},
        )

    def __call__(self, batch: list[LumpedArrayData]) -> LumpedTensorData:
        inputs = SequencedDictArray([v.x_data for v in batch])
        outputs = SequencedDictArray([v.y_data for v in batch])

        # concatenate and send
        inputs_tensors, inputs_batch_info = to_batch(
            inputs.to_phlower_tensors_dict(
                device=self._device,
                non_blocking=self._non_blocking,
                disable_dimensions=self._disable_dimensions,
            ),
            batch_mode_dict=self._batch_mode_holder.inputs_batch_mode,
        )

        outputs_tensors, outputs_batch_info = to_batch(
            outputs.to_phlower_tensors_dict(
                device=self._device,
                non_blocking=self._non_blocking,
                disable_dimensions=self._disable_dimensions,
            ),
            batch_mode_dict=self._batch_mode_holder.labels_batch_mode,
        )

        simulation_field, field_batch_info = _to_batch_field_helper(
            batch=batch,
            device=self._device,
            non_blocking=self._non_blocking,
            disable_dimensions=self._disable_dimensions,
            batch_mode_holder=self._batch_mode_holder,
        )
        data_directories = [b.data_directory for b in batch]

        return LumpedTensorData(
            x_data=phlower_tensor_collection(inputs_tensors),
            y_data=phlower_tensor_collection(outputs_tensors),
            field_data=simulation_field,
            data_directories=data_directories,
            x_batch_info=inputs_batch_info,
            y_batch_info=outputs_batch_info,
            field_batch_info=field_batch_info,
        )


def _to_batch_field_helper(
    batch: list[LumpedArrayData],
    device: str | torch.device,
    non_blocking: bool,
    disable_dimensions: bool,
    batch_mode_holder: BatchModeHolder,
) -> tuple[ISimulationField, GraphBatchInfo]:
    _contain_only_arrays = all(
        all(isinstance(v, IPhlowerArray) for v in d.field_data.values())
        for d in batch
    )
    if _contain_only_arrays:
        _arrays = SequencedDictArray([v.field_data for v in batch])
        _tensors, _batch_info = to_batch(
            _arrays.to_phlower_tensors_dict(
                device=device,
                non_blocking=non_blocking,
                disable_dimensions=disable_dimensions,
            ),
            batch_mode_dict=batch_mode_holder.field_batch_mode,
        )
        return (
            create_simulation_field(
                field_tensors=_tensors,
                batch_info=_batch_info,
            ),
            _batch_info,
        )

    if len(batch) != 1:
        raise NotImplementedError(
            "Batch size must be 1 when field data contains mesh data."
        )

    mesh_data = [
        v
        for v in batch[0].field_data.values()
        if not isinstance(v, IPhlowerArray)
    ]
    assert len(mesh_data) == 1, "Only one mesh data is supported in field data."

    _arrays = {
        name: v
        for name, v in batch[0].field_data.items()
        if isinstance(v, IPhlowerArray)
    }
    _arrays = SequencedDictArray([_arrays])
    _tensors, _batch_info = to_batch(
        _arrays.to_phlower_tensors_dict(
            device=device,
            non_blocking=non_blocking,
            disable_dimensions=disable_dimensions,
        ),
        batch_mode_dict=batch_mode_holder.field_batch_mode,
    )

    return (
        create_simulation_field(
            field_tensors=_tensors,
            batch_info=_batch_info,
            tensor_mesh=mesh_data[0],
            disable_dimensions=disable_dimensions,
        ),
        _batch_info,
    )
