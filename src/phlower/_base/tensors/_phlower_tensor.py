from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from pipe import select

from phlower._base._dimension import PhysicalDimensions
from phlower._base.tensors._dimension_tensor import (
    PhlowerDimensionTensor,
    phlower_dimension_tensor,
)
from phlower._base.tensors._interface import IPhlowerTensor
from phlower.utils import get_logger
from phlower.utils.exceptions import DimensionIncompatibleError

logger = get_logger(__name__)


def phlower_tensor(
    tensor: torch.Tensor | PhlowerTensor,
    dimension: (
        PhysicalDimensions
        | PhlowerDimensionTensor
        | torch.Tensor
        | dict[str, float]
        | None
    ) = None,
    is_time_series: bool = False,
    is_voxel: bool = False,
):
    if isinstance(tensor, PhlowerTensor):
        if dimension is not None:
            logger.warning(
                "Input dimension_tensor and sparse_batch_info are ignored."
            )

        return tensor

    dimension_tensor = _resolve_dimension_arg(dimension)

    return PhlowerTensor(
        tensor=tensor, dimension_tensor=dimension_tensor,
        is_time_series=is_time_series, is_voxel=is_voxel)


def _resolve_dimension_arg(
    inputs: PhysicalDimensions
    | PhlowerDimensionTensor
    | torch.Tensor
    | dict[str, float]
    | None,
) -> PhlowerDimensionTensor | None:
    if inputs is None:
        return None

    if isinstance(inputs, PhlowerDimensionTensor):
        return inputs

    if isinstance(inputs, torch.Tensor):
        return PhlowerDimensionTensor(inputs)

    if isinstance(inputs, dict | PhysicalDimensions):
        return phlower_dimension_tensor(inputs)

    raise NotImplementedError(
        f"{type(inputs)} is not implemented "
        "when creating PhlowerDimensionTensor"
    )


class PhlowerTensor(IPhlowerTensor):
    """PhlowerTensor

    Tensor object which can be with physics dimenstion tensor.

    """

    def __init__(
        self,
        tensor: torch.Tensor,
        dimension_tensor: PhlowerDimensionTensor | None = None,
        is_time_series: bool = False,
        is_voxel: bool = False,
    ):
        assert isinstance(tensor, torch.Tensor)
        self._tensor = tensor
        self._dimension_tensor = dimension_tensor
        self._is_time_series = is_time_series
        self._is_voxel = is_voxel

    @property
    def has_dimension(self) -> bool:
        return self._dimension_tensor is not None

    @property
    def dimension(self) -> PhlowerDimensionTensor | None:
        return self._dimension_tensor

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    @property
    def is_sparse(self) -> bool:
        return self._tensor.layout == torch.sparse_coo

    @property
    def is_time_series(self) -> bool:
        return self._is_time_series

    @property
    def is_voxel(self) -> bool:
        return self._is_voxel

    def __str__(self) -> str:
        return (
            f"PhysicsTensor({self._tensor}, "
            f"Dimension: {self._dimension_tensor})"
        )

    def __abs__(self) -> PhlowerTensor:
        return torch.abs(self)

    def __sub__(self, other: PhlowerTensor):
        return torch.subtract(self, other)

    def __add__(self, other) -> PhlowerTensor:
        return torch.add(self, other)

    def __mul__(self, other) -> PhlowerTensor:
        return torch.mul(self, other)

    def __len__(self) -> int:
        return len(self._tensor)

    def __getitem__(self, key: Any) -> PhlowerTensor:
        return PhlowerTensor(self._tensor[key], self._dimension_tensor)

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def coalesce(self) -> torch.Tensor:
        return PhlowerTensor(self._tensor.coalesce(), self._dimension_tensor)

    def size(self) -> torch.Size:
        return self._tensor.size()

    def rank(self) -> int:
        """Returns the tensor rank."""
        if self.is_sparse:
            raise NotImplementedError
        size = self.size()
        start = 1
        if self.is_time_series:
            start += 1
        if self.is_voxel:
            start += 2
        return len(size[start:-1])

    def indices(self) -> torch.Tensor:
        return self._tensor.indices()

    def values(self) -> torch.Tensor:
        return self._tensor.values()

    def reshape(self, shape: Sequence[int]) -> PhlowerTensor:
        self._tensor = self._tensor.reshape(shape)
        return self

    def slice(self, slice_range: tuple[slice, ...]) -> PhlowerTensor:
        tmp = self._tensor[slice_range]
        return PhlowerTensor(tmp, dimension_tensor=self._dimension_tensor)

    def to(self, device: str, non_blocking: bool = False) -> None:
        self._tensor.to(device, non_blocking=non_blocking)
        if self.has_dimension:
            self._dimension_tensor.to(device, non_blocking=non_blocking)

    def detach(self) -> PhlowerTensor:
        return PhlowerTensor(
            self._tensor.detach(),
            dimension_tensor=self._dimension_tensor.detach(),
        )

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

        if not _has_dimension(args):
            # Unit calculation is not considered when unit tensor is not found.
            return PhlowerTensor(ret)

        _dimensions = _recursive_resolve(
            args, "_dimension_tensor", allow_none=False
        )

        result_units = func(*_dimensions, **kwargs)
        return PhlowerTensor(ret, result_units)


def _recursive_resolve(
    args: Iterable | Any, attr: str, allow_none: bool = True
) -> list[str]:
    if isinstance(args, tuple | list):
        return [
            _recursive_resolve(v, attr, allow_none=allow_none) for v in args
        ]

    _val = getattr(args, attr, args)

    if allow_none:
        return _val

    if _val is None:
        if args is None:
            # If args does not have attribute, treat as it is.
            # For example, bias may be None when calculating torch.nn.Linear
            return None

        raise DimensionIncompatibleError(
            "Cannnot calcualte PhlowerTensor with physics dimension "
            f"and PhlowerTensor without it. {args}"
        )
    return _val


def _has_dimension(args: Any) -> bool:
    if isinstance(args, tuple | list):
        return any(list(args | select(lambda x: _has_dimension(x))))

    if isinstance(args, dict):
        return _has_dimension(list(args.values()))

    if isinstance(args, PhlowerTensor):
        return args.has_dimension

    return False
