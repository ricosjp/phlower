import abc

import ignite.utils as ignite_utils
import torch

from phlower.utils.typing import ArrayDataType, DenseArrayType, SparseArrayType


class IPhlowerArrayCollections(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_tensor(self, device: str, non_blocking: bool):
        raise NotImplementedError()

    @abc.abstractmethod
    def concatenate(self):
        raise NotImplementedError()


class IPhlowerDictArray(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def is_sparse(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self, key: str) -> ArrayDataType:
        raise NotImplementedError()


class PhlowerDictDenseArray(IPhlowerDictArray):
    def __init__(self, data: dict[str, DenseArrayType]) -> None:
        self._data = data

    def to_tensor(self, device: str | torch.device | None, non_blocking: bool):
        return ignite_utils.convert_tensor(
            self._data, device=device, non_blocking=non_blocking
        )

    def __getitem__(self, key: str) -> DenseArrayType:
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()


class PhlowerDictSparseArray(IPhlowerDictArray):
    def __init__(self, data: dict[str, SparseArrayType]) -> None:
        self._data = data

    def to_tensor(self, device: str | torch.device | None, non_blocking: bool):
        torch.sparse_coo_tensor()
        return ignite_utils.convert_tensor(
            self._data, device=device, non_blocking=non_blocking
        )

    def __getitem__(self, key: str) -> SparseArrayType:
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()
