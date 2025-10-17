from __future__ import annotations

import abc
from collections import defaultdict
from collections.abc import (
    Callable,
    ItemsView,
    KeysView,
    Sequence,
    ValuesView,
)

import numpy as np
import torch

from phlower._base import IPhlowerArray, PhlowerTensor, phlower_tensor
from phlower.utils.typing import ArrayDataType


class IPhlowerTensorCollections(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(
        self, value: dict[str, torch.Tensor | PhlowerTensor]
    ) -> None: ...

    @abc.abstractmethod
    def __add__(self, __value: object) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def __sub__(self, __value: object) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def __mul__(self, __value: object) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def __truediv__(self, __value: object) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def __contains__(self, key: str): ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def slice(
        self, slice_range: tuple[slice, ...]
    ) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def __getitem__(self, key: int | str) -> PhlowerTensor: ...

    @abc.abstractmethod
    def send(self, device: str) -> None: ...

    @abc.abstractmethod
    def min_len(self) -> int: ...

    @abc.abstractmethod
    def to_numpy(self) -> dict[str, ArrayDataType]: ...

    @abc.abstractmethod
    def keys(self) -> KeysView[str]: ...

    @abc.abstractmethod
    def values(self): ...

    @abc.abstractmethod
    def items(self) -> ItemsView[str, PhlowerTensor]: ...

    @abc.abstractmethod
    def pop(self, key: str, default: PhlowerTensor | None = None): ...

    @abc.abstractmethod
    def sum(self, weights: dict[str, float] = None) -> PhlowerTensor: ...

    @abc.abstractmethod
    def mean(self, weights: dict[str, float] = None) -> PhlowerTensor: ...

    @abc.abstractmethod
    def reshape(self, shape: Sequence[int]) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def unique_item(self) -> PhlowerTensor: ...

    @abc.abstractmethod
    def update(
        self, data: IPhlowerTensorCollections, overwrite: bool = False
    ) -> None: ...

    @abc.abstractmethod
    def mask(self, keys: list[str]) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def apply(self, function: Callable) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def clone(self) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def snapshot(self, time_index: int) -> IPhlowerTensorCollections:
        """
        Create a snapshot of the current state of the collection
          at a given time index. Timeseries tensor is converted to
          a tensor with the specified time index by index access.
          Non-timeseries tensors are returned as is.

        Args:
            time_index (int): The index of the time point.
        Returns:
            IPhlowerTensorCollections: A new collection containing
            the state of the tensors at the specified time index.
        """
        ...

    @abc.abstractmethod
    def to_phlower_arrays_dict(self) -> dict[str, IPhlowerArray]: ...

    @abc.abstractmethod
    def get_time_series_length(self) -> int:
        """
        Get the length of the time series in the collection.
        Returns:
            int: The length of the time series.
        """
        ...

    @abc.abstractmethod
    def to(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> IPhlowerTensorCollections: ...


def phlower_tensor_collection(
    values: dict[str, torch.Tensor | PhlowerTensor],
) -> IPhlowerTensorCollections:
    if isinstance(values, dict):
        return PhlowerDictTensors(values)

    raise NotImplementedError(
        f"input data type: {type(values)} cannot be converted."
    )


class PhlowerDictTensors(IPhlowerTensorCollections):
    def __init__(self, value: dict[str, torch.Tensor | PhlowerTensor]):
        self._data = {k: phlower_tensor(v) for k, v in value.items()}

    def __lt__(self, __value: object) -> bool:
        if isinstance(__value, PhlowerDictTensors):
            _check_same_keys("compare", self.keys(), __value.keys())
            return all(self._data[k] < __value[k] for k in self.keys())

        return all(self._data[k] < __value for k in self.keys())

    def __le__(self, __value: object) -> bool:
        if isinstance(__value, PhlowerDictTensors):
            _check_same_keys("compare", self.keys(), __value.keys())
            return all(self._data[k] <= __value[k] for k in self.keys())

        return all(self._data[k] <= __value for k in self.keys())

    def __gt__(self, __value: object) -> bool:
        if isinstance(__value, PhlowerDictTensors):
            _check_same_keys("compare", self.keys(), __value.keys())
            return all(self._data[k] > __value[k] for k in self.keys())

        return all(self._data[k] > __value for k in self.keys())

    def __ge__(self, __value: object) -> bool:
        if isinstance(__value, PhlowerDictTensors):
            _check_same_keys("compare", self.keys(), __value.keys())
            return all(self._data[k] >= __value[k] for k in self.keys())

        return all(self._data[k] >= __value for k in self.keys())

    def __add__(self, __value: object) -> IPhlowerTensorCollections:
        if isinstance(__value, PhlowerDictTensors):
            _check_same_keys("add", self.keys(), __value.keys())
            return PhlowerDictTensors(
                {k: self._data[k] + __value[k] for k in self.keys()}
            )

        return PhlowerDictTensors(
            {k: self._data[k] + __value for k in self.keys()}
        )

    def __sub__(self, __value: object) -> IPhlowerTensorCollections:
        if isinstance(__value, PhlowerDictTensors):
            _check_same_keys("substract", self.keys(), __value.keys())
            return PhlowerDictTensors(
                {k: self._data[k] - __value[k] for k in self.keys()}
            )
        return PhlowerDictTensors(
            {k: self._data[k] - __value for k in self.keys()}
        )

    def __mul__(self, __value: object) -> IPhlowerTensorCollections:
        if isinstance(__value, PhlowerDictTensors):
            _check_same_keys("multiple", self.keys(), __value.keys())
            return PhlowerDictTensors(
                {k: self._data[k] * __value[k] for k in self.keys()}
            )
        return PhlowerDictTensors(
            {k: self._data[k] * __value for k in self.keys()}
        )

    def __truediv__(self, __value: object) -> IPhlowerTensorCollections:
        if isinstance(__value, PhlowerDictTensors):
            _check_same_keys("divide", self.keys(), __value.keys())
            return PhlowerDictTensors(
                {k: self._data[k] / __value[k] for k in self.keys()}
            )
        return PhlowerDictTensors(
            {k: self._data[k] / __value for k in self.keys()}
        )

    def __str__(self) -> str:
        txt = ""
        for k, v in self._data.items():
            txt += f"{k}: {v}, "
        return f"{self.__class__.__name__}" + "{" + txt + "}"

    def __contains__(self, key: str):
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def values(self) -> ValuesView[PhlowerTensor]:
        return self._data.values()

    def keys(self) -> KeysView[str]:
        return self._data.keys()

    def items(self) -> ItemsView[str, PhlowerTensor]:
        return self._data.items()

    def pop(
        self, key: str, default: PhlowerTensor | None = None
    ) -> PhlowerTensor | None:
        return self._data.pop(key, default)

    def __getitem__(self, key: str) -> PhlowerTensor:
        if isinstance(key, str):
            return self._data[key]

        raise NotImplementedError(f"key must be str. Input: {type(key)}")

    def reshape(self, shape: Sequence[int]) -> IPhlowerTensorCollections:
        return phlower_tensor_collection(
            {k: v.reshape(shape) for k, v in self._data.items()}
        )

    def slice(
        self, slice_range: tuple[slice, ...]
    ) -> IPhlowerTensorCollections:
        tmp = {k: v.slice(slice_range) for k, v in self._data.items()}
        return phlower_tensor_collection(tmp)

    def send(self, device: str) -> None:
        self._data = {k: v.send(device) for k, v in self._data.items()}

    def min_len(self) -> int:
        return np.min([len(x) for x in self._data.values()])

    def to_numpy(self) -> dict[str, ArrayDataType]:
        return {
            k: v.to_tensor().detach().cpu().numpy() for k, v in self.items()
        }

    def sum(self, weights: dict[str, float] | None = None) -> PhlowerTensor:
        if weights is None:
            return torch.sum(torch.stack(list(self._data.values())))

        return torch.sum(
            torch.stack([v * weights[k] for k, v in self._data.items()])
        )

    def mean(self, weights: dict[str, float] | None = None) -> PhlowerTensor:
        if weights is None:
            return torch.mean(torch.stack(list(self._data.values())))

        total_weight = sum(weights.values())
        return (
            torch.sum(
                torch.stack([v * weights[k] for k, v in self._data.items()])
            )
            / total_weight
        )

    def unique_item(self) -> PhlowerTensor:
        if len(self) != 1:
            raise ValueError(
                "This PhlowerCollections have several items. not unique."
            )

        key = list(self.keys())[0]
        return self[key]

    def update(
        self, data: IPhlowerTensorCollections, overwrite: bool = False
    ) -> None:
        for k in data.keys():
            if (k in self) and (not overwrite):
                raise ValueError(
                    f"{k} already exists. "
                    "If you want to overwrite it, use overwrite=True"
                )

            self._data[k] = data[k]

    def mask(self, keys: list[str]) -> IPhlowerTensorCollections:
        return PhlowerDictTensors({k: self._data[k] for k in keys})

    def apply(self, function: Callable) -> IPhlowerTensorCollections:
        return PhlowerDictTensors(
            {k: function(self._data[k]) for k in self.keys()}
        )

    def clone(self) -> IPhlowerTensorCollections:
        return PhlowerDictTensors({k: v.clone() for k, v in self.items()})

    def to_phlower_arrays_dict(self) -> dict[str, IPhlowerArray]:
        return {k: v.to_phlower_array() for k, v in self.items()}

    def snapshot(self, time_index: int) -> IPhlowerTensorCollections:
        results = {
            k: v.slice_time(time_index, keep_time_series=False)
            if v.is_time_series
            else v
            for k, v in self.items()
        }
        return PhlowerDictTensors(results)

    def get_time_series_length(self) -> int:
        values = {
            v.time_series_length for v in self.values() if v.is_time_series
        }

        if len(values) > 1:
            raise ValueError(
                "Cannot determine time series length, "
                "because there are different time series lengths "
                "in the collection."
            )

        return values.pop() if len(values) == 1 else 0

    def to(
        self, device: str | torch.device, non_blocking: bool = False
    ) -> IPhlowerTensorCollections:
        return phlower_tensor_collection(
            {
                k: v.to(device, non_blocking=non_blocking)
                for k, v in self._data.items()
            }
        )


def reduce_update(
    values: list[IPhlowerTensorCollections],
) -> IPhlowerTensorCollections:
    result = phlower_tensor_collection({})
    for v in values:
        result.update(v)
    return result


def reduce_stack(
    values: list[IPhlowerTensorCollections], to_time_series: bool = False
) -> IPhlowerTensorCollections:
    _results: dict[str, list[PhlowerTensor]] = defaultdict(list)
    for collection in values:
        for k, v in collection.items():
            _results[k].append(v)

    result = {
        k: phlower_tensor(
            torch.stack(v),
            dimension=v[0].dimension,
            is_time_series=to_time_series,
            is_voxel=v[0].is_voxel,
        )
        for k, v in _results.items()
    }
    return phlower_tensor_collection(result)


def _check_same_keys(
    operation_name: str, keys1: KeysView[str], keys2: KeysView[str]
) -> None:
    if keys1 == keys2:
        return

    raise AssertionError(
        f"Not allowed to {operation_name} other which has different keys"
        f"{list(keys1)}, {list(keys2)}"
    )
