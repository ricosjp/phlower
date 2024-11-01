from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeAlias, overload

import einops
import numpy as np
import torch
from pipe import select
from typing_extensions import Self

from phlower._base._dimension import PhysicalDimensions
from phlower._base.tensors._dimension_tensor import (
    PhlowerDimensionTensor,
    phlower_dimension_tensor,
)
from phlower._base.tensors._interface import IPhlowerTensor
from phlower._base.tensors._tensor_shape import PhlowerShapePattern
from phlower.utils import get_logger
from phlower.utils.exceptions import (
    DimensionIncompatibleError,
    PhlowerSparseUnsupportedError,
    PhlowerTypeError,
    PhlowerUnsupportedTorchFunctionError,
)

PhysicDimensionLikeObject: TypeAlias = (
    PhysicalDimensions
    | PhlowerDimensionTensor
    | torch.Tensor
    | dict[str, float]
    | list[float]
    | tuple[float]
)

_UNSUPPORTED_FUNCTION_NAMES = [
    "einsum",
    "reshape",
]


logger = get_logger(__name__)


@overload
def phlower_tensor(
    tensor: list | np.ndarray | torch.Tensor | PhlowerTensor,
    dimension: PhysicDimensionLikeObject | None = None,
    is_time_series: bool = False,
    is_voxel: bool = False,
) -> PhlowerTensor: ...


@overload
def phlower_tensor(
    tensor: list | np.ndarray | torch.Tensor | PhlowerTensor,
    dimension: PhysicDimensionLikeObject | None = None,
    pattern: str = "n...",
) -> PhlowerTensor: ...


def phlower_tensor(
    tensor: list | np.ndarray | torch.Tensor | PhlowerTensor,
    dimension: PhysicDimensionLikeObject | None = None,
    is_time_series: bool | None = None,
    is_voxel: bool | None = None,
    pattern: str | None = None,
):
    if isinstance(tensor, PhlowerTensor):
        if dimension is not None:
            logger.warning("Input dimension_tensor are ignored.")
        return tensor

    if isinstance(tensor, list | np.ndarray):
        tensor = torch.tensor(tensor, dtype=torch.float32)

    dimension_tensor = _resolve_dimension_arg(dimension)

    if pattern is not None:
        if (is_time_series is not None) or (is_voxel is not None):
            raise ValueError(
                "pattern is not allowed to be used "
                "with is_time_series and is_voxel "
            )
        return PhlowerTensor.from_pattern(
            tensor, dimension_tensor=dimension_tensor, pattern=pattern
        )

    return PhlowerTensor(
        tensor=tensor,
        dimension_tensor=dimension_tensor,
        is_time_series=bool(is_time_series),
        is_voxel=bool(is_voxel),
    )


def _resolve_dimension_arg(
    inputs: PhysicalDimensions
    | PhlowerDimensionTensor
    | torch.Tensor
    | dict[str, float]
    | list[float]
    | tuple[float]
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

    if isinstance(inputs, list | tuple):
        return PhlowerDimensionTensor.from_list(inputs)

    raise NotImplementedError(
        f"{type(inputs)} is not implemented "
        "when creating PhlowerDimensionTensor"
    )


class PhlowerTensor(IPhlowerTensor):
    """PhlowerTensor

    Tensor object which can be with physics dimenstion tensor.

    """

    @classmethod
    def from_pattern(
        cls,
        tensor: torch.Tensor,
        dimension_tensor: PhlowerDimensionTensor | None = None,
        pattern: str | PhlowerShapePattern | None = None,
    ) -> PhlowerTensor:
        if pattern is None:
            raise ValueError("pattern must be set when calling from_pattern.")

        phlower_shape: PhlowerShapePattern = PhlowerShapePattern.from_pattern(
            tensor.shape, pattern
        )

        return PhlowerTensor(
            tensor=tensor,
            dimension_tensor=dimension_tensor,
            is_time_series=phlower_shape.is_time_series,
            is_voxel=phlower_shape.is_voxel,
        )

    def __init__(
        self,
        tensor: torch.Tensor,
        dimension_tensor: PhlowerDimensionTensor | None = None,
        is_time_series: bool = False,
        is_voxel: bool = False,
    ):
        if not isinstance(tensor, torch.Tensor):
            raise PhlowerTypeError(
                f"Expect torch.Tensor but {tensor.__class__} was fed"
            )
        self._tensor = tensor
        self._dimension_tensor = dimension_tensor
        self._phlower_shape = PhlowerShapePattern(
            self._tensor.shape, is_time_series=is_time_series, is_voxel=is_voxel
        )

    @property
    def has_dimension(self) -> bool:
        return self._dimension_tensor is not None

    @property
    def dimension(self) -> PhlowerDimensionTensor | None:
        return self._dimension_tensor

    @property
    def shape(self) -> torch.Size:
        return self._phlower_shape.shape

    @property
    def shape_pattern(self) -> PhlowerShapePattern:
        return self._phlower_shape

    @property
    def is_sparse(self) -> bool:
        return self._tensor.layout == torch.sparse_coo

    @property
    def is_time_series(self) -> bool:
        return self._phlower_shape.is_time_series

    @property
    def is_voxel(self) -> bool:
        return self._phlower_shape.is_voxel

    def __repr__(self) -> str:
        return (
            f"PhlowerTensor({self._tensor}, "
            f"Dimension: {self._dimension_tensor}), "
            f"is_time_series: {self.is_time_series}, "
            f"is_voxel: {self.is_voxel}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: PhlowerTensor):
        return torch.eq(self, other)

    def __lt__(self, other: PhlowerTensor):
        return torch.lt(self, other)

    def __le__(self, other: PhlowerTensor):
        return torch.le(self, other)

    def __gt__(self, other: PhlowerTensor):
        return torch.gt(self, other)

    def __ge__(self, other: PhlowerTensor):
        return torch.ge(self, other)

    def __abs__(self) -> PhlowerTensor:
        return torch.abs(self)

    def __sub__(self, other: PhlowerTensor):
        return torch.sub(self, other)

    def __neg__(self):
        return -1 * self

    def __add__(self, other: PhlowerTensor) -> PhlowerTensor:
        return torch.add(self, other)

    def __radd__(self, other: PhlowerTensor) -> PhlowerTensor:
        return torch.add(self, other)

    def __mul__(self, other: PhlowerTensor) -> PhlowerTensor:
        return torch.mul(self, other)

    def __rmul__(self, other: PhlowerTensor) -> PhlowerTensor:
        return torch.mul(self, other)

    def __truediv__(self, other: PhlowerTensor) -> PhlowerTensor:
        return torch.div(self, other)

    def __rtruediv__(self, other: PhlowerTensor) -> PhlowerTensor:
        return torch.div(other, self)

    def __pow__(self, other: PhlowerTensor) -> PhlowerTensor:
        return torch.pow(self, other)

    @functools.wraps(torch.Tensor.__getitem__)
    def __getitem__(self, key: Any) -> torch.Tensor:
        # NOTE: When accessed by index, PhlowerTensor cannot ensure
        # its shape pattern. Thus, return only Tensor object
        return self._tensor[key]

    @functools.wraps(torch.Tensor.__setitem__)
    def __setitem__(self, key: Any, value: Any) -> Self:
        if isinstance(key, PhlowerTensor):
            self._tensor[key.to_tensor()] = value
        else:
            self._tensor[key] = value
        return self

    def __len__(self) -> int:
        return len(self._tensor)

    def to_tensor(self) -> torch.Tensor:
        return self._tensor

    def to_numpy(self) -> np.ndarray:
        return self._tensor.cpu().detach().numpy()

    def numpy(self) -> np.ndarray:
        return self.to_numpy()

    def coalesce(self) -> torch.Tensor:
        return PhlowerTensor(self._tensor.coalesce(), self._dimension_tensor)

    def size(self) -> torch.Size:
        return self._tensor.size()

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    def transpose(self, dim0: int, dim1: int) -> PhlowerTensor:
        _tensor = self._tensor.transpose(dim0, dim1)
        return PhlowerTensor(
            tensor=_tensor,
            dimension_tensor=self._dimension_tensor,
            is_time_series=self.is_time_series,
            is_voxel=self.is_voxel,
        )

    def numel(self) -> int:
        return torch.numel(self._tensor)

    def rank(self) -> int:
        """Returns the tensor rank."""
        if self.is_sparse:
            raise PhlowerSparseUnsupportedError(
                "Cannot call rank() for sparse PhlowerTensor"
            )

        return self._phlower_shape.rank_size

    def n_vertices(self) -> int:
        """Returns the number of vertices."""
        if self.is_sparse:
            raise PhlowerSparseUnsupportedError(
                "Cannot call n_vertices() for sparse PhlowerTensor"
            )

        return self._phlower_shape.get_n_vertices()

    def indices(self) -> torch.Tensor:
        return self._tensor.indices()

    def values(self) -> torch.Tensor:
        return self._tensor.values()

    def to_vertexwise(self) -> tuple[PhlowerTensor, str]:
        """
        Convert to vertexwise 2D tensor which has (n_vertices, -1) shape.

        Returns:
            vertexwise_tensor : PhlowerTensor
                Vertexwise PhlowerTensor object.
            resultant_pattern : str
                Pattern of the resultant shape. Can be used for rearrange.
        """
        original_pattern = self._phlower_shape.get_pattern()
        resultant_pattern = (
            f"({self._phlower_shape.get_space_pattern()}) "
            f"({self._phlower_shape.time_series_pattern} "
            f"{self._phlower_shape.get_feature_pattern()})"
        )
        tensor_2d = einops.rearrange(
            self.to_tensor(), f"{original_pattern} -> {resultant_pattern}"
        )
        return (
            PhlowerTensor.from_pattern(
                tensor_2d,
                dimension_tensor=self.dimension,
                pattern=resultant_pattern,
            ),
            resultant_pattern,
        )

    def rearrange(
        self,
        pattern: str,
        **axes_length: dict[str, int],
    ) -> PhlowerTensor:
        rearranged = einops.rearrange(self._tensor, pattern, **axes_length)

        to_pattern = pattern.split("->")[-1]
        return PhlowerTensor.from_pattern(
            rearranged, dimension_tensor=self.dimension, pattern=to_pattern
        )

    def reshape(
        self,
        shape: Sequence[int],
        is_time_series: bool = False,
        is_voxel: bool = False,
    ) -> PhlowerTensor:
        return PhlowerTensor(
            torch.reshape(self.to_tensor(), shape),
            dimension_tensor=self.dimension,
            is_time_series=is_time_series,
            is_voxel=is_voxel,
        )

    def to(
        self,
        device: str | torch.device = None,
        non_blocking: bool = False,
        dtype: torch.dtype = None,
    ) -> PhlowerTensor:
        new_tensor = self._tensor.to(
            device=device, dtype=dtype, non_blocking=non_blocking
        )
        if self.has_dimension:
            new_dimension = self._dimension_tensor.to(
                device=device, dtype=dtype, non_blocking=non_blocking
            )
        else:
            new_dimension = None
        return PhlowerTensor.from_pattern(
            new_tensor, new_dimension, pattern=self.shape_pattern
        )

    def detach(self) -> PhlowerTensor:
        return PhlowerTensor(
            self._tensor.detach(),
            dimension_tensor=self._dimension_tensor.detach(),
            is_time_series=self.is_time_series,
            is_voxel=self.is_voxel,
        )

    def backward(self) -> None:
        self._tensor.backward()

    def as_pattern(self, pattern: str) -> PhlowerTensor:
        return PhlowerTensor.from_pattern(
            tensor=self._tensor,
            dimension_tensor=self._dimension_tensor,
            pattern=pattern,
        )

    def clone(self) -> PhlowerTensor:
        tensor = self._tensor.clone()
        dimension = PhlowerDimensionTensor(
            self._dimension_tensor._tensor.clone()
        )
        return PhlowerTensor(
            tensor,
            dimension_tensor=dimension,
            is_time_series=self.is_time_series,
            is_voxel=self.is_voxel,
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: list[type],
        args: tuple,
        kwargs: dict | None = None,
    ) -> PhlowerTensor:
        if func.__name__ in _UNSUPPORTED_FUNCTION_NAMES:
            raise PhlowerUnsupportedTorchFunctionError(
                f"Unsupported function: {func.__name__}"
            )
        if kwargs is None:
            kwargs = {}

        _tensors = _recursive_resolve(args, "_tensor")

        ret: torch.Tensor = func(*_tensors, **kwargs)

        # NOTE: Assume flags for the first tensor is preserved
        is_time_series = _recursive_resolve(
            args, "is_time_series", return_first_only=True
        )
        is_voxel = _recursive_resolve(args, "is_voxel", return_first_only=True)

        if not _has_dimension(args):
            # Unit calculation is not considered when unit tensor is not found.
            return PhlowerTensor(
                ret, is_time_series=is_time_series, is_voxel=is_voxel
            )

        _dimensions = _recursive_resolve(
            args, "_dimension_tensor", allow_none=False
        )

        result_units = func(*_dimensions, **kwargs)
        return PhlowerTensor(
            ret, result_units, is_time_series=is_time_series, is_voxel=is_voxel
        )


def _recursive_resolve(
    args: Iterable | Any,
    attr: str,
    allow_none: bool = True,
    return_first_only: bool = False,
) -> list[str]:
    if isinstance(args, tuple | list):
        if return_first_only:
            return _recursive_resolve(
                args[0],
                attr,
                allow_none=allow_none,
                return_first_only=return_first_only,
            )
        else:
            return [
                _recursive_resolve(
                    v,
                    attr,
                    allow_none=allow_none,
                    return_first_only=return_first_only,
                )
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
