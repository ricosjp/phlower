from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import einops
import numpy as np
import torch
from pipe import select

from phlower._base._dimension import PhysicalDimensions
from phlower._base.tensors._dimension_tensor import (
    PhlowerDimensionTensor,
    phlower_dimension_tensor,
)
from phlower._base.tensors._interface import IPhlowerTensor
from phlower._base.tensors._unsupported_function_names import (
    UNSUPPORTED_FUNCTION_NAMES,
)
from phlower.utils import get_logger
from phlower.utils.exceptions import (
    DimensionIncompatibleError,
    PhlowerReshapeError,
    PhlowerSparseUnsupportedError,
    PhlowerUnsupportedTorchFunctionError,
)

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
        original_pattern: str | None = None,
        current_pattern: str | None = None,
        dict_shape: dict[str, int] | None = None,
    ):
        assert isinstance(tensor, torch.Tensor)
        self._tensor = tensor
        self._dimension_tensor = dimension_tensor
        self._is_time_series = is_time_series
        self._is_voxel = is_voxel
        self._original_pattern = original_pattern
        self._current_pattern = current_pattern
        self._dict_shape = dict_shape

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

    @property
    def original_pattern(self) -> str | None:
        return self._original_pattern

    @property
    def current_pattern(self) -> str | None:
        return self._current_pattern

    @property
    def dict_shape(self) -> dict[str, int] | None:
        return self._dict_shape

    def __str__(self) -> str:
        return (
            f"PhlowerTensor({self._tensor}, "
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

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def coalesce(self) -> torch.Tensor:
        return PhlowerTensor(self._tensor.coalesce(), self._dimension_tensor)

    def size(self) -> torch.Size:
        return self._tensor.size()

    def rank(self) -> int:
        """Returns the tensor rank."""
        if self.is_sparse:
            raise PhlowerSparseUnsupportedError(
                "Cannot call rank() for sparse PhlowerTensor")
        size = self.size()
        start = 1
        if self.is_time_series:
            start += 1
        if self.is_voxel:
            start += 2
        return len(size[start:-1])

    def n_vertices(self) -> int:
        """Returns the number of vertices."""
        if self.is_sparse:
            raise PhlowerSparseUnsupportedError(
                "Cannot call n_vertices() for sparse PhlowerTensor")
        size = self.size()
        start = 0
        if self.is_time_series:
            start += 1

        if self.is_voxel:
            return np.prod(size[start:start+3])

        return size[start]

    def indices(self) -> torch.Tensor:
        return self._tensor.indices()

    def values(self) -> torch.Tensor:
        return self._tensor.values()

    def to_2d(self) -> PhlowerTensor:
        """Convert to 2D tensor which has (n_vertices, -1) shape"""
        shape = self.shape
        dict_shape = {}

        space_start = 0
        if self.is_time_series:
            t_pattern = "t "
            space_start += 1
            dict_shape.update({"t": shape[0]})
        else:
            t_pattern = ""
        if self.is_voxel:
            space_pattern = "x y z "
            try:
                dict_shape.update({
                    "x": shape[space_start],
                    "y": shape[space_start + 1],
                    "z": shape[space_start + 2],
                })
            except:
                raise ValueError(self.shape, self.is_time_series, self.is_voxel)
            feat_start = space_start + 3
        else:
            space_pattern = "n "
            dict_shape.update({"n": shape[space_start]})
            feat_start = space_start + 1

        feat_pattern = " ".join([
            f"a{i}" for i in range(len(shape[feat_start:]))])
        # Do not include the last axis in case modified by NNs.
        dict_shape.update({
            f"a{i}": s for i, s in enumerate(shape[feat_start:-1])})

        original_pattern = f"{t_pattern}{space_pattern}{feat_pattern}"
        current_pattern = f"({space_pattern}) ({t_pattern}{feat_pattern})"
        tensor_2d = einops.rearrange(
            self.to_tensor(), f"{original_pattern} -> {current_pattern}")
        return PhlowerTensor(
            tensor_2d, dimension_tensor=self.dimension,
            is_time_series=False, is_voxel=False,
            original_pattern=original_pattern,
            current_pattern=current_pattern,
            dict_shape=dict_shape)

    def from_2d(self) -> PhlowerTensor:
        if self.original_pattern is None:
            raise PhlowerReshapeError(
                "No original_pattern found. Run to_2d first.")
        is_time_series = "t" in self.original_pattern
        is_voxel = "x" in self.original_pattern
        return self.inverse_rearrange(
            is_time_series=is_time_series, is_voxel=is_voxel)

    def rearrange(
            self, pattern: str,
            is_time_series: bool = False, is_voxel: bool = False,
            **kwargs) -> PhlowerTensor:
        tensor = self.to_tensor()
        original_pattern, current_pattern = pattern.split("->")
        rearranged = einops.rearrange(tensor, pattern, **kwargs)
        return PhlowerTensor(
            rearranged, dimension_tensor=self.dimension,
            is_time_series=is_time_series, is_voxel=is_voxel,
            original_pattern=original_pattern, current_pattern=current_pattern,
            dict_shape=kwargs)

    def inverse_rearrange(
            self, is_time_series: bool = False, is_voxel: bool = False,
            **kwargs) -> PhlowerTensor:
        if self.original_pattern is None:
            raise PhlowerReshapeError(
                "No original_pattern found. Run rearrange first.")
        pattern = f"{self.current_pattern} -> {self.original_pattern}"
        kwargs.update(self.dict_shape)
        return self.rearrange(
            pattern, is_time_series=is_time_series, is_voxel=is_voxel,
            **kwargs)

    def reshape(
        self, shape: Sequence[int],
        is_time_series: bool = False, is_voxel: bool = False,
    ) -> PhlowerTensor:
        return PhlowerTensor(
            torch.reshape(self.to_tensor(), shape),
            dimension_tensor=self.dimension,
            is_time_series=is_time_series, is_voxel=is_voxel)

    def to(self, device: str, non_blocking: bool = False) -> None:
        self._tensor.to(device, non_blocking=non_blocking)
        if self.has_dimension:
            self._dimension_tensor.to(device, non_blocking=non_blocking)

    def detach(self) -> PhlowerTensor:
        return PhlowerTensor(
            self._tensor.detach(),
            dimension_tensor=self._dimension_tensor.detach(),
            is_time_series=self.is_time_series, is_voxel=self.is_voxel,
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
        if func.__name__ in UNSUPPORTED_FUNCTION_NAMES:
            raise PhlowerUnsupportedTorchFunctionError(
                f"Unsupported function: {func.__name__}")
        if kwargs is None:
            kwargs = {}

        _tensors = _recursive_resolve(args, "_tensor")

        ret: torch.Tensor = func(*_tensors, **kwargs)

        # NOTE: Assume flags for the first tensor is preserved
        is_time_series = _recursive_resolve(
            args, "_is_time_series", return_first_only=True)
        is_voxel = _recursive_resolve(
            args, "_is_voxel", return_first_only=True)

        if not _has_dimension(args):
            # Unit calculation is not considered when unit tensor is not found.
            return PhlowerTensor(
                ret, is_time_series=is_time_series, is_voxel=is_voxel)

        _dimensions = _recursive_resolve(
            args, "_dimension_tensor", allow_none=False
        )

        result_units = func(*_dimensions, **kwargs)
        return PhlowerTensor(
            ret, result_units, is_time_series=is_time_series,
            is_voxel=is_voxel)


def _recursive_resolve(
    args: Iterable | Any, attr: str, allow_none: bool = True,
    return_first_only: bool = False,
) -> list[str]:
    if isinstance(args, tuple | list):
        if return_first_only:
            return _recursive_resolve(
                args[0], attr, allow_none=allow_none,
                return_first_only=return_first_only)
        else:
            return [
                _recursive_resolve(
                    v, attr, allow_none=allow_none,
                    return_first_only=return_first_only)
                for v in args
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
