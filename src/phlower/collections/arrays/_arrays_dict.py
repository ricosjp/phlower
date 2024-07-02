from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from phlower._base.array import IPhlowerArray, phlower_array, sparse


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

    def concatenate(self) -> IPhlowerArray:
        if len(self) == 1:
            return self._data[0]

        if self._is_sparse:
            return sparse.batch(self._data)

        if self._is_timeseries:
            return phlower_array(np.concatenate(self._data, axis=1))

        return phlower_array(np.concatenate(self._data, axis=0))


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

    def concatenate(self) -> dict[str, IPhlowerArray]:
        return {
            name: arr.concatenate()
            for name, arr in self._phlower_sequece_dict.items()
        }
