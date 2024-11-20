from __future__ import annotations

import abc
from _collections_abc import dict_items
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch

from phlower._base import PhlowerTensor, phlower_tensor
from phlower.utils.typing import ArrayDataType


class IPhlowerTensorCollections(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(
        self, value: dict[str, torch.Tensor | PhlowerTensor]
    ) -> None: ...

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
    def keys(self) -> Iterable[str]: ...

    @abc.abstractmethod
    def values(self): ...

    @abc.abstractmethod
    def pop(self, key: str, default: PhlowerTensor | None = None): ...

    @abc.abstractmethod
    def sum(self, weights: dict[str, float] = None) -> PhlowerTensor: ...

    @abc.abstractmethod
    def reshape(self, shape: Sequence[int]) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def unique_item(self) -> PhlowerTensor: ...

    @abc.abstractmethod
    def update(self, data: IPhlowerTensorCollections) -> None: ...


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

    def values(self) -> Iterable[dict[str, Any]]:
        return self._x.values()

    def keys(self) -> Iterable[str]:
        return self._x.keys()

    def items(self) -> dict_items[str, PhlowerTensor]:
        return self._x.items()

    def pop(
        self, key: str, default: PhlowerTensor | None = None
    ) -> PhlowerTensor | None:
        return self._x.pop(key, default)

    def __getitem__(self, key: str) -> PhlowerTensor:
        if isinstance(key, str):
            return self._x[key]

        raise NotImplementedError(f"key must be str. Input: {type(key)}")

    def reshape(self, shape: Sequence[int]) -> IPhlowerTensorCollections:
        return phlower_tensor_collection(
            {k: v.reshape(shape) for k, v in self._x.items()}
        )

    def slice(
        self, slice_range: tuple[slice, ...]
    ) -> IPhlowerTensorCollections:
        tmp = {k: v.slice(slice_range) for k, v in self._x.items()}
        return phlower_tensor_collection(tmp)

    def send(self, device: str) -> None:
        self._x = {k: v.send(device) for k, v in self._x.items()}

    def min_len(self) -> int:
        return np.min([len(x) for x in self._x.values()])

    def to_numpy(self) -> dict[str, ArrayDataType]:
        return {
            k: v.to_tensor().detach().cpu().numpy() for k, v in self.items()
        }

    def sum(self, weights: dict[str, float] = None) -> PhlowerTensor:
        if weights is None:
            return torch.sum(torch.stack(list(self._x.values())))

        return torch.sum(
            torch.stack([v * weights[k] for k, v in self._x.items()])
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

            self._x[k] = data[k]


def reduce_collections(
    values: list[IPhlowerTensorCollections],
) -> IPhlowerTensorCollections:
    result = phlower_tensor_collection({})
    for v in values:
        result.update(v)
    return result
