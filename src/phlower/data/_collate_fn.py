from collections.abc import Sequence

import ignite
import numpy as np
import torch
from ignite import utils

from phlower.collections.tensors import phlower_tensor_collection, IPhlowerTensorCollections
from phlower.base.array import IPhlowerArray, phlower_arrray
from phlower.base.tensors import PhlowerTensor
from phlower.data._bunched_data import BunchedData, BunchedTensorData
from phlower.utils.typing import ArrayDataType


class SequencedDictArray:
    def __init__(self, data: list[dict[str, IPhlowerArray]]) -> None:
        self._data = data

    def get_names(self) -> Sequence[str]:
        return self._data[0].keys()

    def concatenate(self) -> dict[str, IPhlowerArray]:
        return {name: self._concatenate(name) for name in self.get_names()}

    def _concatenate(self, name: str) -> IPhlowerArray:
        assert len(self._data) == 1
        return self._data[0][name]
        # TODO: Under construction
        # change how to concatenate ndarray
        # according to properties such as time_series, sparse
        # return np.concatenate([v[name] for v in self._data])


def _to_tensor(
    dict_data: dict[str, IPhlowerArray],
    device: str | torch.device,
    non_blocking: bool = False,
    dimensions: dict[str, dict[str, float]] = None
) -> IPhlowerTensorCollections:
    return phlower_tensor_collection(
        {
            k: v.to_phlower_tensor(
                device=device, non_blocking=non_blocking, dimension=dimensions.get(k)
            )
            for k, v in dict_data.items()
        }
    )


class PhlowerCollateFn:
    def __init__(self, device: str | torch.device, non_blocking: bool = False, dimensions: dict[str, dict[str, float]] = None) -> None:
        self._device = device
        self._non_blocking = non_blocking
        self._dimensions = dimensions

    def __call__(self, batch: list[BunchedData]) -> BunchedTensorData:

        inputs = SequencedDictArray([v.x_data for v in batch])
        outputs = SequencedDictArray([v.y_data for v in batch])
        sparse_supports = SequencedDictArray([v.sparse_supports for v in batch])

        # concatenate and send
        inputs = _to_tensor(
            inputs.concatenate(),
            device=self._device,
            non_blocking=self._non_blocking,
            dimensions=self._dimensions
        )
        outputs = _to_tensor(
            outputs.concatenate(),
            device=self._device,
            non_blocking=self._non_blocking,
            dimensions=self._dimensions
        )
        sparse_supports = _to_tensor(
            sparse_supports.concatenate(),
            device=self._device,
            non_blocking=self._non_blocking,
            dimensions=self._dimensions
        )
        data_directories = [b.data_directory for b in batch]

        return BunchedTensorData(
            x_data=inputs,
            y_data=outputs,
            sparse_supports=sparse_supports,
            data_directory=data_directories,
        )
