from __future__ import annotations

import abc
from collections.abc import Callable

import torch

from phlower._base.tensors import PhlowerDimensionTensor


class IPhlowerTensor(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def has_dimension(self) -> bool: ...

    @abc.abstractproperty
    def dimension(self) -> PhlowerDimensionTensor | None: ...

    @abc.abstractproperty
    def shape(self) -> torch.Size: ...

    @abc.abstractproperty
    def is_sparse(self) -> bool: ...

    @abc.abstractproperty
    def is_time_series(self) -> bool: ...

    @abc.abstractproperty
    def is_voxel(self) -> bool: ...

    @abc.abstractmethod
    def values(self) -> torch.Tensor: ...

    @abc.abstractmethod
    def __str__(self) -> str: ...

    @abc.abstractmethod
    def __add__(self, other) -> IPhlowerTensor: ...

    @abc.abstractmethod
    def __mul__(self, other) -> IPhlowerTensor: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def to_tensor(self) -> torch.Tensor: ...

    @abc.abstractmethod
    def size(self) -> torch.Size: ...

    @abc.abstractmethod
    def rank(self) -> int: ...

    @abc.abstractmethod
    def n_vertices(self) -> int: ...

    @abc.abstractmethod
    def indices(self) -> torch.Tensor: ...

    @abc.abstractmethod
    def coalesce(self) -> IPhlowerTensor: ...

    @abc.abstractmethod
    def to(self, device: str, non_blocking: bool = False) -> None: ...

    @abc.abstractmethod
    def backward(self) -> None: ...

    @abc.abstractclassmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: list[type],
        args: tuple,
        kwargs: dict | None = None,
    ): ...
