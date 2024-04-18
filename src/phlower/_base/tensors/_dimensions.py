from __future__ import annotations

import enum
import functools
from typing import Callable, Union

import torch

from phlower.utils.exceptions import DimensionIncompatibleError

_HANDLED_FUNCTIONS: dict[str, Callable] = {}


class PhysicalDimensionType(enum.Enum):
    mass = 0  # Ex: kilogram
    length = 1  # Ex: metre
    time = 2  # Ex: second
    thermodynamic_temperature = 3  # Ex: Kelvin
    amount_of_substance = 4  # Ex: mole
    electric_current = 5  # Ex. ampere
    luminous_intensity = 6  # Ex. candela


def phlower_dimension_tensor(
    values: dict[str, float], dtype: torch.dtype = torch.float32
) -> PhlowerDimensionTensor:
    _list: list[float] = [0 for _ in range(len(PhysicalDimensionType))]
    for k, v in values.items():
        if k not in PhysicalDimensionType.__members__:
            raise NotImplementedError(
                f"dimension name: {k} is not implemented."
                f"Avaliable units are {list(PhysicalDimensionType)}"
            )
        _list[PhysicalDimensionType[k].value] = v

    return PhlowerDimensionTensor.from_list(_list)


def zero_dimension_tensor() -> PhlowerDimensionTensor:
    return phlower_dimension_tensor({})


class PhlowerDimensionTensor:
    @classmethod
    def from_list(
        cls, values: list[float] | tuple[float]
    ) -> PhlowerDimensionTensor:
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
            self._tensor = torch.zeros(
                (len(PhysicalDimensionType), 1), dtype=dtype
            )
            return

        self._tensor = tensor
        assert self._tensor.shape[0] == len(PhysicalDimensionType)

    def __add__(self, __value: object):
        return torch.add(self, __value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhlowerDimensionTensor):
            return NotImplemented

        return bool(torch.all(self._tensor == other._tensor).item())

    def __pow__(self, other: float | int) -> PhlowerDimensionTensor:
        return PhlowerDimensionTensor(tensor=self._tensor * other)

    def __repr__(self) -> str:
        texts = ", ".join(
            [
                f"{unit_type.name}: {self._tensor[unit_type.value, 0].item()}"
                for unit_type in PhysicalDimensionType
            ]
        )
        return f"{self.__class__.__name__}({texts})"

    def to(self, device: str | torch.device, non_blocking: bool = False):
        self._tensor.to(device, non_blocking=non_blocking)

    @classmethod
    def __torch_function__(cls, func, types, args: tuple, kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented

        return _HANDLED_FUNCTIONS[func](*args, **kwargs)


def dimension_wrap_implements(torch_function):
    """Register a torch function override for PhysicsUnitTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        _HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@dimension_wrap_implements(torch.mean)
def mean(inputs: PhlowerDimensionTensor):
    return PhlowerDimensionTensor(inputs._tensor)


@dimension_wrap_implements(torch.add)
def add(inputs, other):
    if all((isinstance(v, PhlowerDimensionTensor) for v in (inputs, other))):
        if inputs != other:
            raise DimensionIncompatibleError()

        return PhlowerDimensionTensor(inputs._tensor)

    raise DimensionIncompatibleError()


@dimension_wrap_implements(torch.mul)
def mul(inputs, other):
    _input = (
        inputs
        if isinstance(inputs, PhlowerDimensionTensor)
        else zero_dimension_tensor()
    )
    _other = (
        other
        if isinstance(other, PhlowerDimensionTensor)
        else zero_dimension_tensor()
    )
    return PhlowerDimensionTensor(_input._tensor + _other._tensor)


@dimension_wrap_implements(torch.reshape)
def reshape(inputs, shape):
    return PhlowerDimensionTensor(inputs._tensor)


@dimension_wrap_implements(torch.cat)
def cat(
    tensors: Union[PhlowerDimensionTensor, torch.Tensor],
    dim: int,
    *,
    out: PhlowerDimensionTensor = None,
) -> PhlowerDimensionTensor:
    if all((isinstance(v, PhlowerDimensionTensor) for v in tensors)):

        # HACK: is it possible to use unique method ?
        for v in tensors:
            if v != tensors[0]:
                raise DimensionIncompatibleError()
    return PhlowerDimensionTensor(tensors[0]._tensor)


@dimension_wrap_implements(torch.sparse.mm)
def sparse_mm(inputs, other):
    if all((isinstance(v, PhlowerDimensionTensor) for v in (inputs, other))):
        return PhlowerDimensionTensor(inputs._tensor + other._tensor)

    raise DimensionIncompatibleError(f"input1: {inputs}, input2: {other}")


@dimension_wrap_implements(torch.nn.functional.linear)
def nn_linear(inputs, *args):
    return inputs


@dimension_wrap_implements(torch.nn.functional.dropout)
def dropout(inputs, *args, **kwards):
    return inputs


@dimension_wrap_implements(torch.stack)
def stack(inputs):
    if all((isinstance(v, PhlowerDimensionTensor) for v in inputs)):

        # HACK: is it possible to use unique method ?
        for v in inputs:
            if v != inputs[0]:
                raise DimensionIncompatibleError()

        return PhlowerDimensionTensor(inputs[0]._tensor)

    raise DimensionIncompatibleError()


@dimension_wrap_implements(torch.nn.functional.mse_loss)
def mse_loss(inputs, others, *args, **kwards):
    if inputs != others:
        raise DimensionIncompatibleError()

    return inputs**2


@dimension_wrap_implements(torch.sum)
def _sum(inputs, *args, **kwards):
    if isinstance(inputs, PhlowerDimensionTensor):
        return inputs

    raise DimensionIncompatibleError()
