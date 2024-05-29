import torch

from phlower._base.array import IPhlowerArray
from phlower.collections.arrays import SequencedDictArray
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.data._lumped_data import LumpedArrayData, LumpedTensorData


def _to_tensor(
    dict_data: dict[str, IPhlowerArray],
    device: str | torch.device,
    non_blocking: bool = False,
    dimensions: dict[str, dict[str, float]] = None,
) -> IPhlowerTensorCollections:
    if dimensions is None:
        dimensions = {}

    return phlower_tensor_collection(
        {
            k: v.to_phlower_tensor(
                device=device,
                non_blocking=non_blocking,
                dimension=dimensions.get(k),
            )
            for k, v in dict_data.items()
        }
    )


class PhlowerCollateFn:
    def __init__(
        self,
        device: str | torch.device,
        non_blocking: bool = False,
        dimensions: dict[str, dict[str, float]] = None,
    ) -> None:
        self._device = device
        self._non_blocking = non_blocking
        self._dimensions = dimensions

    def __call__(self, batch: list[LumpedArrayData]) -> LumpedTensorData:

        inputs = SequencedDictArray([v.x_data for v in batch])
        outputs = SequencedDictArray([v.y_data for v in batch])
        sparse_supports = SequencedDictArray([v.sparse_supports for v in batch])

        # concatenate and send
        inputs = _to_tensor(
            inputs.concatenate(),
            device=self._device,
            non_blocking=self._non_blocking,
            dimensions=self._dimensions,
        )
        outputs = _to_tensor(
            outputs.concatenate(),
            device=self._device,
            non_blocking=self._non_blocking,
            dimensions=self._dimensions,
        )
        sparse_supports = _to_tensor(
            sparse_supports.concatenate(),
            device=self._device,
            non_blocking=self._non_blocking,
            dimensions=self._dimensions,
        )
        data_directories = [b.data_directory for b in batch]

        return LumpedTensorData(
            x_data=inputs,
            y_data=outputs,
            sparse_supports=sparse_supports,
            data_directory=data_directories,
        )
