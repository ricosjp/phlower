from __future__ import annotations

import abc
from collections.abc import Callable

import torch

from phlower._base.tensors import PhlowerDimensionTensor
from phlower._base.tensors._tensor_shape import PhlowerTensorShapePattern


class IPhlowerTensor(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def has_dimension(self) -> bool: ...

    @property
    @abc.abstractmethod
    def dimension(self) -> PhlowerDimensionTensor | None: ...

    @property
    @abc.abstractmethod
    def shape(self) -> torch.Size: ...

    @property
    @abc.abstractmethod
    def shape_pattern(self) -> PhlowerTensorShapePattern: ...

    @property
    @abc.abstractmethod
    def is_sparse(self) -> bool: ...

    @property
    @abc.abstractmethod
    def is_time_series(self) -> bool: ...

    @property
    @abc.abstractmethod
    def is_voxel(self) -> bool: ...

    @abc.abstractmethod
    def values(self) -> torch.Tensor: ...

    @abc.abstractmethod
    def __str__(self) -> str: ...

    @abc.abstractmethod
    def __add__(self, other: IPhlowerTensor) -> IPhlowerTensor: ...

    @abc.abstractmethod
    def __mul__(self, other: IPhlowerTensor) -> IPhlowerTensor: ...

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

    @abc.abstractmethod
    def to_vertexwise(self) -> tuple[IPhlowerTensor, str]: ...

    @abc.abstractmethod
    def rearrange(
        self,
        pattern: str,
        is_time_series: bool = False,
        is_voxel: bool = False,
        **kwargs: dict[str, int],
    ) -> IPhlowerTensor: ...

    @abc.abstractmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: list[type],
        args: tuple,
        kwargs: dict | None = None,
    ): ...
