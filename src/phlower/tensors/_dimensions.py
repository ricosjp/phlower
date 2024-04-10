from __future__ import annotations

import enum
import functools
from typing import Callable

import torch

from phlower.utils.exceptions import UnitIncompatibleError

_HANDLED_FUNCTIONS: dict[str, Callable] = {}


class PhysicalDimensionType(enum.Enum):
    kg = 0
    m = 1
    s = 2
    K = 3
    mol = 4
    A = 5
    Cd = 6


def physical_dimension_tensor(
    values: dict[str, float], dtype: torch.dtype = torch.float32
) -> PhlowerDimensionTensor:
    _list: list[float] = [0 for _ in range(len(PhysicalDimensionType))]
    for k, v in values.items():
        if k not in PhysicalDimensionType.__members__:
            raise NotImplementedError(
                f"unit name: {k} is not implemented."
                f"Avaliable units are {list(PhysicalDimensionType)}"
            )
        _list[PhysicalDimensionType[k].value] = v

    return PhlowerDimensionTensor.create(_list)


class PhlowerDimensionTensor:
    @classmethod
    def create(cls, values: list[float] | tuple[float]) -> PhlowerDimensionTensor:
        assert len(values) == len(PhysicalDimensionType)
        _tensor = torch.tensor(values, dtype=torch.int64).reshape(
            len(PhysicalDimensionType), 1
        )
        return PhlowerDimensionTensor(_tensor)

    def __init__(
        self,
        tensor: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if tensor is None:
            self._tensor = torch.zeros((len(PhysicalDimensionType), 1), dtype=dtype)
            return

        self._tensor = tensor
        assert self._tensor.shape[0] == len(PhysicalDimensionType)

    def __add__(self, __value: object):
        return torch.add(self, __value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhlowerDimensionTensor):
            return NotImplemented

        return bool(torch.all(self._tensor == other._tensor).item())

    def __repr__(self) -> str:
        texts = ", ".join(
            [
                f"{unit_type.name}: {self._tensor[unit_type.value, 0].item()}"
                for unit_type in PhysicalDimensionType
            ]
        )
        return f"{self.__class__.__name__}, Unit Dimension Info. {texts}"

    @classmethod
    def __torch_function__(cls, func, types, args: tuple, kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented

        return _HANDLED_FUNCTIONS[func](*args, **kwargs)


def unit_wrap_implements(torch_function):
    """Register a torch function override for PhysicsUnitTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        _HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@unit_wrap_implements(torch.mean)
def mean(inputs: PhlowerDimensionTensor):
    return PhlowerDimensionTensor(inputs._tensor)


@unit_wrap_implements(torch.add)
def add(inputs, other):
    if all((isinstance(v, PhlowerDimensionTensor) for v in (inputs, other))):
        if inputs != other:
            raise UnitIncompatibleError()

        return PhlowerDimensionTensor(inputs._tensor)

    raise UnitIncompatibleError()


@unit_wrap_implements(torch.mul)
def mul(inputs, other):
    if all((isinstance(v, PhlowerDimensionTensor) for v in (inputs, other))):
        return PhlowerDimensionTensor(inputs._tensor + other._tensor)

    raise UnitIncompatibleError()
