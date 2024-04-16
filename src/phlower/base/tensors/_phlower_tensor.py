from __future__ import annotations
import abc

from collections.abc import Sequence
from typing import Callable, Any

import numpy as np
import torch

from ._dimensions import PhlowerDimensionTensor


class PhlowerTensor:

    def __init__(
        self,
        tensor: torch.Tensor,
        unit_tensor: torch.Tensor | None = None,
        shapes: list[tuple[int]] = None
    ):
        if isinstance(tensor, PhlowerTensor):
            self._copy(tensor)
            return

        assert isinstance(tensor, torch.Tensor)
        self._tensor = tensor
        self._unit_tensor = unit_tensor
        self._is_time_series: bool = False  # HACK: UNDER CONSTRUCTION
        self._shapes = shapes if shapes is not None else [tensor.shape]

    def _copy(self, other: PhlowerTensor) -> None:
        self._tensor = other._tensor
        self._unit_tensor = other._unit_tensor
        self._shapes = other._shapes
        self._is_time_series = other._is_time_series

    @property
    def has_unit(self) -> bool:
        return self._unit_tensor is not None

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    def __str__(self) -> str:
        return f"PhysicsTensor({self._tensor})"

    def __add__(self, other) -> PhlowerTensor:
        return torch.add(self, other)

    def __mul__(self, other) -> PhlowerTensor:
        return torch.mul(self, other)

    def __len__(self) -> int:
        return len(self._tensor)

    def tensor(self) -> torch.Tensor:
        return self._tensor

    def size(self) -> torch.Size:
        return self._tensor.size()

    def reshape(self, shape: Sequence[int]) -> PhlowerTensor:
        self._tensor = self._tensor.reshape(shape)
        return self

    def slice(self, slice_range: tuple[slice, ...]) -> PhlowerTensor:
        tmp = self._tensor[slice_range]
        return PhlowerTensor(tmp, unit_tensor=self._unit_tensor)

    def send(self, device: str) -> None:
        self._tensor.to(device)

    def split(self) -> list[torch.Tensor]:
        if len(self._shapes) == 1:
            return [self._tensor]

        # HACK: NEED TO IMPLEMENT
        # for v1, v2 in zip(self._shapes[:-1], self._shapes[1:]):
        #     ...
        raise NotImplementedError()

    def backward(self) -> None:
        self._tensor.backward()

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: list[type],
        args: tuple,
        kwargs: dict | None = None,
    ):
        if kwargs is None:
            kwargs = {}

        _tensors = _recursive_resolve(args, "_tensor")
        ret: torch.Tensor = func(*_tensors, **kwargs)

        _units = _recursive_resolve(args, "_unit_tensor")

        if not all((isinstance(v, PhlowerDimensionTensor) for v in _units)):
            # Unit calculation is not considered when unit tensor is not found.
            return PhlowerTensor(ret)

        result_units = func(*_units, **kwargs)
        return PhlowerTensor(ret, result_units)


def _recursive_resolve(args: Any, attr: str = None) -> list[str]:
    if isinstance(args, (list, tuple)):
        return [_recursive_resolve(v, attr) for v in args]

    return getattr(args, attr, args)
