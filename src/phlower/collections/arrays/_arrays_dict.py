from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch

from phlower._base import GraphBatchInfo, IPhlowerArray, PhlowerTensor
from phlower._base._functionals import to_batch


class _PhlowerSequenceArray:
    def __init__(self, name: str, data: list[IPhlowerArray]) -> None:
        self._name = name
        self._data = data
        assert len(self._data) > 0

        self._is_sparse = self._reduce_is_sparse()
        self._is_timeseries = self._reduce_is_timeseries()

    def _reduce_is_sparse(self):
        _is_sparse = np.unique(np.array([v.is_sparse for v in self._data]))
        if len(_is_sparse) != 1:
            raise ValueError(
                "Sparse array and dense array are mixed "
                f"in the same variable name; {self._name}"
            )
        return _is_sparse.item()

    def _reduce_is_timeseries(self):
        _is_time_series = np.unique(
            np.array([v.is_time_series for v in self._data])
        )
        if len(_is_time_series) != 1:
            raise ValueError(
                "time-series array and steady array are mixed "
                f"in the same variable name; {self._name}"
            )
        return _is_time_series.item()

    def __len__(self) -> int:
        return len(self._data)

    @property
    def is_timeseries(self) -> bool:
        return self._is_timeseries

    @property
    def is_sparse(self) -> bool:
        return self._is_sparse

    def to_batched_tensor(
        self,
        device: str | torch.device | None = None,
        non_blocking: bool = False,
        dimensions: dict[str, float] | None = None,
    ):
        tensors = [
            v.to_phlower_tensor(
                device=device, non_blocking=non_blocking, dimension=dimensions
            )
            for v in self._data
        ]

        if self.is_sparse:
            return to_batch(tensors)

        if self.is_timeseries:
            return to_batch(tensors, dense_concat_dim=1)

        return to_batch(tensors, dense_concat_dim=0)


class SequencedDictArray:
    def __init__(self, data: list[dict[str, IPhlowerArray]]) -> None:
        _sequece_array_dict: dict[str, list[IPhlowerArray]] = defaultdict(list)

        for dict_data in data:
            for k, v in dict_data.items():
                _sequece_array_dict[k].append(v)

        for k, v in _sequece_array_dict.items():
            if len(v) != len(data):
                raise ValueError(
                    f"the number of data is not consistent in {k}. "
                    f"Please check that {k} exists in all data"
                )

        self._phlower_sequece_dict = {
            k: _PhlowerSequenceArray(k, arr)
            for k, arr in _sequece_array_dict.items()
        }

    def get_names(self) -> Sequence[str]:
        return self._phlower_sequece_dict.keys()

    def to_batched_tensor(
        self,
        device: str,
        non_blocking: bool,
        dimensions: dict[str, dict[str, float]] | None = None,
    ) -> tuple[dict[str, PhlowerTensor], dict[str, GraphBatchInfo]]:
        _batched = [
            (
                name,
                arr.to_batched_tensor(
                    device=device,
                    non_blocking=non_blocking,
                    dimensions=dimensions.get(name),
                ),
            )
            for name, arr in self._phlower_sequece_dict.items()
        ]
        _tensors = {k: v[0] for k, v in _batched}
        _batched_info = {k: v[1] for k, v in _batched}

        return _tensors, _batched_info
