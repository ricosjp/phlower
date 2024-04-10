from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ._dimensions import PhlowerDimensionTensor


class PhlowerTensor:

    def __init__(
        self,
        tensor: torch.Tensor | np.ndarray,
        unit_tensor: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self._tensor: torch.Tensor
        if not isinstance(tensor, torch.Tensor):
            self._tensor = torch.tensor(tensor, dtype=dtype)
        else:
            self._tensor = tensor

        self._is_time_series: bool = False
        self._unit_tensor = unit_tensor

    @property
    def has_unit(self) -> bool:
        return self._unit_tensor is not None

    def __str__(self) -> str:
        return f"PhysicsTensor: {self._tensor}"

    def __add__(self, other) -> PhlowerTensor:
        return torch.add(self, other)

    def __mul__(self, other) -> PhlowerTensor:
        return torch.mul(self, other)

    def tensor(self) -> torch.Tensor:
        return self._tensor

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

        _tensors = [getattr(a, "_tensor", a) for a in args]
        ret: torch.Tensor = func(*_tensors, **kwargs)

        _units = [getattr(a, "_unit_tensor", a) for a in args]

        if not all((isinstance(v, PhlowerDimensionTensor) for v in _units)):
            # Unit calculation is not considered when unit tensor is not found.
            return PhlowerTensor(ret)

        result_units = func(*_units, **kwargs)
        return PhlowerTensor(ret, result_units)
