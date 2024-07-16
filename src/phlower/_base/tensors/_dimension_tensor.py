from __future__ import annotations

import functools
from collections.abc import Callable

import torch

from phlower._base._dimension import PhysicsDimensions
from phlower.utils.enums import PhysicalDimensionType
from phlower.utils.exceptions import DimensionIncompatibleError

_HANDLED_FUNCTIONS: dict[str, Callable] = {}


def phlower_dimension_tensor(
    values: dict[str, float] | PhysicsDimensions, dtype: torch.dtype = torch.float32
) -> PhlowerDimensionTensor:

    if not isinstance(values, PhysicsDimensions):
        values = PhysicsDimensions(values)

    _list = values.to_list()
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

    def to_physics_dimension(self):
        _dict = self.to_dict()
        return PhysicsDimensions(_dict)

    def to_dict(self) -> dict[str, float]:
        return {
            k: self._tensor[v.value].numpy().item()
            for k, v in PhysicalDimensionType.__members__.items()
        }

    def to(self, device: str | torch.device, non_blocking: bool = False):
        self._tensor.to(device, non_blocking=non_blocking)

    def detach(self) -> PhlowerDimensionTensor:
        return PhlowerDimensionTensor(tensor=self._tensor.detach())

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
    if all(isinstance(v, PhlowerDimensionTensor) for v in (inputs, other)):
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
    tensors: tuple[PhlowerDimensionTensor | torch.Tensor, ...],
    dim: int,
    *,
    out: PhlowerDimensionTensor = None,
) -> PhlowerDimensionTensor:
    if all(isinstance(v, PhlowerDimensionTensor) for v in tensors):
        # HACK: is it possible to use unique method ?
        for v in tensors:
            if v != tensors[0]:
                raise DimensionIncompatibleError()
    return PhlowerDimensionTensor(tensors[0]._tensor)


@dimension_wrap_implements(torch.sparse.mm)
def sparse_mm(inputs, other):
    if all(isinstance(v, PhlowerDimensionTensor) for v in (inputs, other)):
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
    if all(isinstance(v, PhlowerDimensionTensor) for v in inputs):
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


@dimension_wrap_implements(torch.concatenate)
def concatenate(inputs, *args, **kwards):
    if all(isinstance(v, PhlowerDimensionTensor) for v in inputs):
        # HACK: is it possible to use unique method ?
        for v in inputs:
            if v != inputs[0]:
                raise DimensionIncompatibleError()

        return PhlowerDimensionTensor(inputs[0]._tensor)

    raise DimensionIncompatibleError()
