from __future__ import annotations
import abc

from typing import Iterable
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
import torch

from phlower.utils.typing import ArrayDataType
from phlower.base.tensors import PhlowerTensor, phlower_tensor


class IPhlowerTensorCollections(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, value: dict[str, torch.Tensor | PhlowerTensor]) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def __contains__(self, key: str):
        raise NotImplementedError()

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def slice(self, slice_range: tuple[slice, ...]) -> IPhlowerTensorCollections:
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self, key: int | str) -> PhlowerTensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def send(self, device: str) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def min_len(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_numpy(self) -> dict[str, ArrayDataType]:
        raise NotImplementedError()

    @abc.abstractmethod
    def keys(self) -> Iterable[str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def values(self):
        ...

    @abc.abstractmethod
    def sum(self, weights: dict[str, float] = None) -> PhlowerTensor:
        raise NotImplementedError()


def phlower_tensor_collection(
    values: Mapping
) -> IPhlowerTensorCollections:
 
    if isinstance(values, dict):
        return PhlowerDictTensors(values)

    raise NotImplementedError(
        f"input data type: {type(values)} is not implemented"
    )


class PhlowerDictTensors(IPhlowerTensorCollections):
    def __init__(self, value: dict[str, torch.Tensor | PhlowerTensor]):
        self._x = {k: phlower_tensor(v) for k, v in value.items()}

    def __str__(self) -> str:
        txt = ""
        for k, v in self._x.items():
            txt += f"{k}: {v}, "
        return f"{self.__class__.__name__}" + "{" + txt + "}"

    def __contains__(self, key: str):
        return key in self._x

    def __len__(self) -> int:
        return len(self._x)

    def values(self):
        return self._x.values()

    def keys(self) -> Iterable[str]:
        return self._x.keys()

    def __getitem__(self, key: Any) -> PhlowerTensor:
        if isinstance(key, str):
            return self._x[key]

        raise NotImplementedError(f"key must be str. Input: {type(key)}")

    def reshape(self, shape: Sequence[int]) -> IPhlowerTensorCollections:
        return  phlower_tensor_collection({k: v.reshape(shape) for k, v in self._x.items()})

    def slice(self, slice_range: tuple[slice, ...]) -> IPhlowerTensorCollections:
        tmp = {k: v.slice(slice_range) for k, v in self._x.items()}
        return phlower_tensor_collection(tmp)

    def send(self, device: str) -> None:
        self._x = {k: v.send(device) for k, v in self._x.items()}

    def min_len(self) -> int:
        return np.min([len(x) for x in self._x.values()])

    def to_numpy(self) -> dict[str, ArrayDataType]:
        return {k: v.tensor().detach().cpu().numpy() for k, v in self._x.items()}

    def sum(self, weights: dict[str, float] = None) -> PhlowerTensor:
        if weights is None:
            return torch.sum(torch.stack([v for v in self._x.values()]))

        return torch.sum(torch.stack([v * weights[k] for k, v in self._x.items()]))

    def unique_item(self) -> PhlowerTensor:
        if len(self) != 1:
            raise ValueError("This PhlowerCollections have several items. not unique.")

        key = list(self.keys())[0]
        return self[key]
