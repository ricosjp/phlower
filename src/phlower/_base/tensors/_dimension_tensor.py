from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

from phlower._base._dimension import PhysicalDimensions
from phlower.utils.enums import PhysicalDimensionSymbolType
from phlower.utils.exceptions import DimensionIncompatibleError

_HANDLED_FUNCTIONS: dict[str, Callable] = {}


def phlower_dimension_tensor(
    values: dict[str, float] | PhysicalDimensions,
    dtype: torch.dtype = torch.float32,
) -> PhlowerDimensionTensor:
    if not isinstance(values, PhysicalDimensions):
        values = PhysicalDimensions(values)

    _list = values.to_list()
    return PhlowerDimensionTensor.from_list(_list)


def zero_dimension_tensor() -> PhlowerDimensionTensor:
    return phlower_dimension_tensor({})


class PhlowerDimensionTensor:
    """
    PhlowerDimensionTensor

    Tensor object which corresponds to physics dimension.
    """

    @classmethod
    def from_list(
        cls, values: list[float] | tuple[float]
    ) -> PhlowerDimensionTensor:
        """
        Parse from list object

        Args:
            values (list[float] | tuple[float]): list or tuple
                if length of values is not equal to the number of registered
                dimension type, raise ValueError.

        Returns:
            PhlowerDimensionTensor: tensor object
        """
        if len(values) != len(PhysicalDimensionSymbolType):
            raise ValueError(
                "length of values is not equal to the number of "
                f"registered dimension types. input: {len(values)}"
            )
        _tensor = torch.tensor(values, dtype=torch.int64).reshape(
            len(PhysicalDimensionSymbolType), 1
        )
        return PhlowerDimensionTensor(_tensor)

    def __init__(
        self,
        tensor: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if tensor is None:
            self._tensor = torch.zeros(
                (len(PhysicalDimensionSymbolType), 1), dtype=dtype
            )
            return

        self._tensor = tensor
        assert self._tensor.shape[0] == len(PhysicalDimensionSymbolType)

    def __add__(self, __value: object):
        return torch.add(self, __value)

    def __mul__(self, __value: object):
        return torch.mul(self, __value)

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
                for unit_type in PhysicalDimensionSymbolType
            ]
        )
        return f"{self.__class__.__name__}({texts})"

    def to_physics_dimension(self) -> PhysicalDimensions:
        _dict = self.to_dict()
        return PhysicalDimensions(_dict)

    def to_dict(self) -> dict[str, float]:
        return {
            k: self._tensor[v.value].numpy().item()
            for k, v in PhysicalDimensionSymbolType.__members__.items()
        }

    def to(
        self,
        device: str | torch.device = None,
        non_blocking: bool = False,
        dtype: torch.dtype = None,
    ) -> PhlowerDimensionTensor:
        new_dimension = self._tensor.to(
            device, non_blocking=non_blocking, dtype=dtype
        )
        new_dtype = dtype or self.dtype
        return PhlowerDimensionTensor(new_dimension, dtype=new_dtype)

    def detach(self) -> PhlowerDimensionTensor:
        return PhlowerDimensionTensor(tensor=self._tensor.detach())

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    @property
    def is_dimensionless(self) -> bool:
        """Return True if the tensor is dimensionless."""
        return torch.sum(torch.abs(self._tensor)) < 1e-5

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: list[type],
        args: tuple,
        kwargs: dict | None = None,
    ) -> PhlowerDimensionTensor:
        if kwargs is None:
            kwargs = {}

        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented

        return _HANDLED_FUNCTIONS[func](*args, **kwargs)


def dimension_wrap_implements(torch_function: Callable) -> Callable:
    """Register a torch function override for PhysicsUnitTensor"""

    def decorator(func: Callable) -> Callable:
        functools.update_wrapper(func, torch_function)
        _HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@dimension_wrap_implements(torch.mean)
def mean(inputs: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(inputs._tensor)


@dimension_wrap_implements(torch.add)
def add(
    inputs: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    if all(isinstance(v, PhlowerDimensionTensor) for v in (inputs, other)):
        if inputs != other:
            raise DimensionIncompatibleError(
                "Add operation for different physical dimensions is not "
                "allowed."
            )

        return PhlowerDimensionTensor(inputs._tensor)

    if all(
            isinstance(v, float) or v.is_dimensionless
            for v in (inputs, other)
    ):
        for v in (inputs, other):
            if isinstance(v, PhlowerDimensionTensor):
                device = v.device
                break
        return zero_dimension_tensor().to(device)

    raise DimensionIncompatibleError()


@dimension_wrap_implements(torch.sub)
def sub(
    inputs: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    if all(isinstance(v, PhlowerDimensionTensor) for v in (inputs, other)):
        if inputs != other:
            raise DimensionIncompatibleError(
                "Sub operation for different physical dimensions is not "
                "allowed."
            )

        return PhlowerDimensionTensor(inputs._tensor)

    if all(
            isinstance(v, float) or v.is_dimensionless
            for v in (inputs, other)
    ):
        for v in (inputs, other):
            if isinstance(v, PhlowerDimensionTensor):
                device = v.device
                break
        return zero_dimension_tensor().to(device)

    raise DimensionIncompatibleError()


@dimension_wrap_implements(torch.pow)
def pow(
    inputs: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(inputs._tensor * other)


@dimension_wrap_implements(torch.sqrt)
def sqrt(inputs: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(inputs._tensor / 2)


@dimension_wrap_implements(torch.mul)
def mul(
    inputs: PhlowerDimensionTensor | torch.Tensor,
    other: PhlowerDimensionTensor | torch.Tensor,
) -> PhlowerDimensionTensor:
    _input = (
        inputs
        if isinstance(inputs, PhlowerDimensionTensor)
        else zero_dimension_tensor().to(other.device)
    )
    _other = (
        other
        if isinstance(other, PhlowerDimensionTensor)
        else zero_dimension_tensor().to(inputs.device)
    )
    return PhlowerDimensionTensor(_input._tensor + _other._tensor)


@dimension_wrap_implements(torch.div)
def div(
    inputs: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    _input = (
        inputs
        if isinstance(inputs, PhlowerDimensionTensor)
        else zero_dimension_tensor().to(other.device)
    )
    _other = (
        other
        if isinstance(other, PhlowerDimensionTensor)
        else zero_dimension_tensor().to(inputs.device)
    )
    return PhlowerDimensionTensor(_input._tensor - _other._tensor)


@dimension_wrap_implements(torch.reshape)
def reshape(
    inputs: PhlowerDimensionTensor, shape: tuple[int]
) -> PhlowerDimensionTensor:
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
def sparse_mm(
    inputs: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    if all(isinstance(v, PhlowerDimensionTensor) for v in (inputs, other)):
        return PhlowerDimensionTensor(inputs._tensor + other._tensor)

    raise DimensionIncompatibleError(f"input1: {inputs}, input2: {other}")


@dimension_wrap_implements(torch.nn.functional.linear)
def nn_linear(
    inputs: PhlowerDimensionTensor, *args: Any
) -> PhlowerDimensionTensor:
    return inputs


@dimension_wrap_implements(torch.nn.functional.dropout)
def dropout(
    inputs: PhlowerDimensionTensor, *args: Any, **kwards: Any
) -> PhlowerDimensionTensor:
    return inputs


@dimension_wrap_implements(torch.stack)
def stack(inputs: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    if all(isinstance(v, PhlowerDimensionTensor) for v in inputs):
        # HACK: is it possible to use unique method ?
        for v in inputs:
            if v != inputs[0]:
                raise DimensionIncompatibleError()

        return PhlowerDimensionTensor(inputs[0]._tensor)

    raise DimensionIncompatibleError()


@dimension_wrap_implements(torch.nn.functional.mse_loss)
def mse_loss(
    inputs: PhlowerDimensionTensor,
    others: PhlowerDimensionTensor,
    *args: Any,
    **kwards: Any,
) -> PhlowerDimensionTensor:
    if inputs != others:
        raise DimensionIncompatibleError()

    return inputs**2


@dimension_wrap_implements(torch.sum)
def _sum(
    inputs: PhlowerDimensionTensor, *args: Any, **kwards: Any
) -> PhlowerDimensionTensor:
    if isinstance(inputs, PhlowerDimensionTensor):
        return inputs

    raise DimensionIncompatibleError()


@dimension_wrap_implements(torch.concatenate)
def concatenate(
    inputs: PhlowerDimensionTensor, *args: Any, **kwards: Any
) -> PhlowerDimensionTensor:
    if all(isinstance(v, PhlowerDimensionTensor) for v in inputs):
        # HACK: is it possible to use unique method ?
        for v in inputs:
            if v != inputs[0]:
                raise DimensionIncompatibleError()

        return PhlowerDimensionTensor(inputs[0]._tensor)

    raise DimensionIncompatibleError()


@dimension_wrap_implements(torch.tanh)
def tanh(tensor: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    if not tensor.is_dimensionless:
        raise DimensionIncompatibleError(
            f"Should be dimensionless to apply tanh but {tensor}"
        )
    return tensor


@dimension_wrap_implements(torch.nn.functional.leaky_relu)
def leaky_relu(
    tensor: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    if not tensor.is_dimensionless:
        raise DimensionIncompatibleError(
            f"Should be dimensionless to apply leaky_relu but {tensor}"
        )
    return tensor


@dimension_wrap_implements(torch.unsqueeze)
def unsqueeze(
    input: PhlowerDimensionTensor, dim: int
) -> PhlowerDimensionTensor:
    return input
