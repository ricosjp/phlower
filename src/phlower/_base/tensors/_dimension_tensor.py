from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from pipe import uniq

from phlower._base._dimension import PhysicalDimensions
from phlower.utils.enums import PhysicalDimensionSymbolType
from phlower.utils.exceptions import DimensionIncompatibleError

_HANDLED_FUNCTIONS: dict[str, Callable] = {}


def phlower_dimension_tensor(
    values: dict[str, float] | PhysicalDimensions,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = None,
) -> PhlowerDimensionTensor:
    """Create PhlowerDimensionTensor from dict or PhysicalDimensions object

    Args:
        values (dict[str, float] | PhysicalDimensions): values
        dtype (torch.dtype, optional): data type. Defaults to torch.float32.
        device (str | torch.device, optional): device. Defaults to None.

    Returns:
        PhlowerDimensionTensor: PhlowerDimensionTensor object
    """
    if not isinstance(values, PhysicalDimensions):
        values = PhysicalDimensions(values)

    _list = values.to_list()
    return PhlowerDimensionTensor.from_list(_list, dtype=dtype).to(device)


def zero_dimension_tensor(
    device: str | torch.device | None,
) -> PhlowerDimensionTensor:
    zerodim = phlower_dimension_tensor({})
    if device is None:
        return zerodim

    return zerodim.to(device)


class PhlowerDimensionTensor:
    """
    PhlowerDimensionTensor is a tensor that represents the physical dimensions

    Examples
    --------
    >>> dimension_tensor = PhlowerDimensionTensor(
    ...     values={"M": 1.0, "L": 1.0, "T": -2.0}
    ... )
    """

    @classmethod
    def from_list(
        cls,
        values: list[float] | tuple[float],
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
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
        _tensor = (
            torch.tensor(values, dtype=dtype)
            .reshape(len(PhysicalDimensionSymbolType), 1)
            .to(device=device)
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

        assert isinstance(tensor, torch.Tensor), f"Unexpected type: {tensor}"

        self._tensor = tensor
        assert self._tensor.shape[0] == len(PhysicalDimensionSymbolType)

    def __sub__(self, __value: object):
        return torch.sub(self, __value)

    def __rsub__(self, __value: object):
        return torch.sub(self, __value)

    def __add__(self, __value: object):
        return torch.add(self, __value)

    def __radd__(self, __value: object):
        return torch.add(self, __value)

    def __mul__(self, __value: object):
        return torch.mul(self, __value)

    def __rmul__(self, __value: object):
        return torch.mul(self, __value)

    def __truediv__(self, __value: object):
        return torch.div(self, __value)

    def __rtruediv__(self, __value: object):
        return torch.div(__value, self)

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
        """Convert to PhysicalDimensions object

        Returns:
            PhysicalDimensions: PhysicalDimensions object
        """
        _dict = self.to_dict()
        return PhysicalDimensions(_dict)

    def to_dict(self) -> dict[str, float]:
        """Convert to dict

        Returns:
            dict[str, float]: Dict
        """
        return {
            k: self._tensor[v.value].cpu().numpy().item()
            for k, v in PhysicalDimensionSymbolType.__members__.items()
        }

    def to(
        self,
        device: str | torch.device = None,
        non_blocking: bool = False,
        dtype: torch.dtype = None,
    ) -> PhlowerDimensionTensor:
        """Convert to different device or data type

        Args:
            device (str | torch.device, optional): device. Defaults to None.
            non_blocking (bool, optional): non-blocking. Defaults to False.
            dtype (torch.dtype, optional): data type. Defaults to None.
        """
        new_dimension = self._tensor.to(
            device, non_blocking=non_blocking, dtype=dtype
        )
        new_dtype = dtype or self.dtype
        return PhlowerDimensionTensor(new_dimension, dtype=new_dtype)

    def detach(self) -> PhlowerDimensionTensor:
        return PhlowerDimensionTensor(tensor=self._tensor.detach())

    def clone(self) -> PhlowerDimensionTensor:
        return PhlowerDimensionTensor(tensor=self._tensor.clone())

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    @property
    def is_dimensionless(self) -> bool:
        """Return True if the tensor is dimensionless.

        Returns:
            bool: True if the tensor is dimensionless.
        """
        return torch.sum(torch.abs(self._tensor)) < 1e-5

    def numpy(self) -> np.ndarray:
        """Convert to numpy array

        Returns:
            np.ndarray: numpy array
        """
        return self._tensor.detach().numpy()

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
def mean(
    inputs: PhlowerDimensionTensor, *args: Any, **kwards: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(inputs._tensor)


def _determine_device(
    *args: PhlowerDimensionTensor | torch.Tensor | float | int,
) -> torch.device:
    devices = {
        v.device
        for v in args
        if isinstance(v, PhlowerDimensionTensor | torch.Tensor)
    }

    if len(devices) != 1:
        raise PhlowerDimensionTensor(
            f"Cannot determine unique device. {devices}. args: {args}"
        )

    return devices.pop()


def _convert_phlower_dimension_tensors(
    *args: PhlowerDimensionTensor | torch.Tensor | float | int,
    device: str | torch.device | None = None,
) -> tuple[PhlowerDimensionTensor, ...]:
    _dimensions = (
        v
        if isinstance(v, PhlowerDimensionTensor)
        else zero_dimension_tensor(device=device)
        for v in args
    )
    return _dimensions


@dimension_wrap_implements(torch.add)
def add(
    inputs: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    device = _determine_device(inputs, other)
    inputs, other = _convert_phlower_dimension_tensors(
        inputs, other, device=device
    )
    if inputs != other:
        raise DimensionIncompatibleError(
            "Add operation for different physical dimensions is not " "allowed."
        )

    return PhlowerDimensionTensor(inputs._tensor)


@dimension_wrap_implements(torch.sub)
def sub(
    inputs: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    device = _determine_device(inputs, other)
    inputs, other = _convert_phlower_dimension_tensors(
        inputs, other, device=device
    )
    if inputs != other:
        raise DimensionIncompatibleError(
            "Sub operation for different physical dimensions is not " "allowed."
        )

    return PhlowerDimensionTensor(inputs._tensor)


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
    device = _determine_device(inputs, other)
    _inputs, _other = _convert_phlower_dimension_tensors(
        inputs, other, device=device
    )
    return PhlowerDimensionTensor(_inputs._tensor + _other._tensor)


@dimension_wrap_implements(torch.div)
def div(
    inputs: PhlowerDimensionTensor, other: PhlowerDimensionTensor
) -> PhlowerDimensionTensor:
    device = _determine_device(inputs, other)
    _inputs, _other = _convert_phlower_dimension_tensors(
        inputs, other, device=device
    )
    return PhlowerDimensionTensor(_inputs._tensor - _other._tensor)


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
    device = _determine_device(*tensors)
    uniq_inputs: list[PhlowerDimensionTensor] = list(
        _convert_phlower_dimension_tensors(*tensors, device=device) | uniq
    )

    if len(uniq_inputs) != 1:
        raise DimensionIncompatibleError(
            "Only same physical dimensions are allowed when torch.cat."
        )

    return PhlowerDimensionTensor(uniq_inputs[0]._tensor)


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
def stack(
    inputs: PhlowerDimensionTensor, *args: Any, **kwards: Any
) -> PhlowerDimensionTensor:
    device = _determine_device(*inputs)
    uniq_inputs: list[PhlowerDimensionTensor] = list(
        _convert_phlower_dimension_tensors(*inputs, device=device) | uniq
    )

    if len(uniq_inputs) != 1:
        raise DimensionIncompatibleError(
            "Only same physical dimensions are allowed when torch.stack"
        )

    return PhlowerDimensionTensor(uniq_inputs[0]._tensor)


@dimension_wrap_implements(torch.max)
def max(
    inputs: PhlowerDimensionTensor, *args: Any, **kwards: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(inputs._tensor)


@dimension_wrap_implements(torch.min)
def min(
    inputs: PhlowerDimensionTensor, *args: Any, **kwards: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(inputs._tensor)


@dimension_wrap_implements(torch.linalg.norm)
def norm(
    inputs: PhlowerDimensionTensor, *args: Any, **kwards: Any
) -> PhlowerDimensionTensor:
    return PhlowerDimensionTensor(inputs._tensor)


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
    device = _determine_device(*inputs)
    uniq_inputs: list[PhlowerDimensionTensor] = list(
        _convert_phlower_dimension_tensors(*inputs, device=device) | uniq
    )

    return PhlowerDimensionTensor(uniq_inputs[0]._tensor)


@dimension_wrap_implements(torch.exp)
def exp(tensor: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    if not tensor.is_dimensionless:
        raise DimensionIncompatibleError(
            f"Should be dimensionless to apply exp but {tensor}"
        )
    return tensor


@dimension_wrap_implements(torch.tanh)
def tanh(tensor: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    if not tensor.is_dimensionless:
        raise DimensionIncompatibleError(
            f"Should be dimensionless to apply tanh but {tensor}"
        )
    return tensor


@dimension_wrap_implements(torch.sigmoid)
def sigmoid(tensor: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    if not tensor.is_dimensionless:
        raise DimensionIncompatibleError(
            f"Should be dimensionless to apply sigmoid but {tensor}"
        )
    return tensor


@dimension_wrap_implements(torch.nn.functional.leaky_relu)
def leaky_relu(
    tensor: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    # NOTE: Allow leaky relu operation also for dimensioned tensor
    #       because it is scale equivariant
    return tensor


@dimension_wrap_implements(torch.unsqueeze)
def unsqueeze(
    inputs: PhlowerDimensionTensor, dim: int
) -> PhlowerDimensionTensor:
    return inputs


@dimension_wrap_implements(torch.abs)
def torch_abs(
    inputs: PhlowerDimensionTensor, *args: Any, **kwards: Any
) -> PhlowerDimensionTensor:
    return inputs


@dimension_wrap_implements(torch.gt)
def torch_gt(
    inputs: PhlowerDimensionTensor,
    others: PhlowerDimensionTensor,
    *args: Any,
    **kwards: Any,
) -> PhlowerDimensionTensor:
    if inputs != others:
        raise DimensionIncompatibleError()
    return inputs


@dimension_wrap_implements(torch.ge)
def torch_ge(
    inputs: PhlowerDimensionTensor,
    others: PhlowerDimensionTensor,
    *args: Any,
    **kwards: Any,
) -> PhlowerDimensionTensor:
    if inputs != others:
        raise DimensionIncompatibleError()
    return inputs


@dimension_wrap_implements(torch.lt)
def torch_lt(
    inputs: PhlowerDimensionTensor,
    others: PhlowerDimensionTensor,
    *args: Any,
    **kwards: Any,
) -> PhlowerDimensionTensor:
    if inputs != others:
        raise DimensionIncompatibleError()
    return inputs


@dimension_wrap_implements(torch.le)
def torch_le(
    inputs: PhlowerDimensionTensor,
    others: PhlowerDimensionTensor,
    *args: Any,
    **kwards: Any,
) -> PhlowerDimensionTensor:
    if inputs != others:
        raise DimensionIncompatibleError()
    return inputs


@dimension_wrap_implements(torch.nn.functional.pad)
def _torch_pad(
    inputs: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return inputs


@dimension_wrap_implements(torch.conv1d)
def _torch_conv1d(
    inputs: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return inputs


@dimension_wrap_implements(torch.relu)
def _torch_relu(inputs: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    # NOTE: Allow relu operation also for dimensioned tensor
    #       because it is scale equivariant
    return inputs


@dimension_wrap_implements(torch.squeeze)
def _torch_squeeze(
    inputs: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return inputs


@dimension_wrap_implements(torch.where)
def _torch_where(
    inputs: PhlowerDimensionTensor, *args: Any, **kwargs: Any
) -> PhlowerDimensionTensor:
    return inputs


@dimension_wrap_implements(torch.sin)
def _torch_sin(inputs: PhlowerDimensionTensor) -> PhlowerDimensionTensor:
    if not inputs.is_dimensionless:
        raise DimensionIncompatibleError(
            f"Should be dimensionless to apply sin but {inputs}"
        )
    return inputs
