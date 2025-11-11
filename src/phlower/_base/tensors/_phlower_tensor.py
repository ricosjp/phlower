from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NamedTuple, TypeAlias, overload

import einops
import numpy as np
import torch
from pipe import select
from typing_extensions import Self

from phlower._base._dimension import PhysicalDimensions
from phlower._base.array import IPhlowerArray, phlower_array
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

_UNSUPPORTED_FUNCTION_NAMES = ["einsum", "reshape", "squeeze"]


logger = get_logger(__name__)


def _is_all_none(*args: Any) -> bool:
    return all(arg is None for arg in args)


@overload
def phlower_tensor(
    tensor: list | float | np.ndarray | torch.Tensor | PhlowerTensor,
    dimension: PhysicDimensionLikeObject | None = None,
    is_time_series: bool = False,
    is_voxel: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> PhlowerTensor: ...


@overload
def phlower_tensor(
    tensor: list | float | np.ndarray | torch.Tensor | PhlowerTensor,
    dimension: PhysicDimensionLikeObject | None = None,
    pattern: str = "n...",
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> PhlowerTensor: ...


def phlower_tensor(
    tensor: list | float | np.ndarray | torch.Tensor | PhlowerTensor,
    dimension: PhysicDimensionLikeObject | None = None,
    is_time_series: bool | None = None,
    is_voxel: bool | None = None,
    pattern: str | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> PhlowerTensor:
    dimension_tensor = _resolve_dimension_arg(
        dimension, dtype=dtype, device=device
    )

    if isinstance(tensor, PhlowerTensor):
        return PhlowerTensor(
            tensor.to_tensor(),
            dimension_tensor=dimension_tensor or tensor.dimension,
            is_time_series=is_time_series or tensor.is_time_series,
            is_voxel=is_voxel or tensor.is_voxel,
        )

    if isinstance(tensor, float | list | np.ndarray):
        tensor = torch.tensor(tensor, dtype=dtype, device=device)

    if pattern is not None:
        if not _is_all_none(is_time_series, is_voxel):
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
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> PhlowerDimensionTensor | None:
    if inputs is None:
        return None

    if isinstance(inputs, PhlowerDimensionTensor):
        return inputs

    if isinstance(inputs, torch.Tensor):
        return PhlowerDimensionTensor(inputs)

    if isinstance(inputs, dict | PhysicalDimensions):
        return phlower_dimension_tensor(inputs, dtype=dtype, device=device)

    if isinstance(inputs, list | tuple):
        return PhlowerDimensionTensor.from_list(
            inputs, dtype=dtype, device=device
        )

    raise NotImplementedError(
        f"{type(inputs)} is not implemented "
        "when creating PhlowerDimensionTensor"
    )


class PhlowerTensor(IPhlowerTensor):
    """
    PhlowerTensor is a wrapper of torch.Tensor with a physical dimension.

    Parameters
    ----------
    tensor (torch.Tensor):
        Tensor to be wrapped.
    dimension_tensor (PhlowerDimensionTensor, optional):
        Physical dimension tensor.
    is_time_series (bool, optional):
        Whether the tensor is a time series.
    is_voxel (bool, optional):
        Whether the tensor is a voxel.

    Examples
    --------
    >>> tensor = torch.randn(5, 100, 3)
    >>> phlower_tensor = PhlowerTensor(
    ...     tensor,
    ...     dimension_tensor={"L": 1, "T": -1},
    ...     is_time_series=True,
    ...     is_voxel=False,
    ... )
    >>> print(phlower_tensor)
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

        phlower_shape = PhlowerShapePattern.from_pattern(tensor.shape, pattern)

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
    def time_series_length(self) -> int:
        """Length of the time series.

        Returns:
            int | None: Length of the time series.
        """
        if not self.is_time_series:
            return 0

        return self._tensor.shape[0]

    @property
    def has_dimension(self) -> bool:
        """Whether the tensor has a physical dimension.

        Returns:
            bool: Whether the tensor has a physical dimension.
        """
        return self._dimension_tensor is not None

    @property
    def dimension(self) -> PhlowerDimensionTensor | None:
        """Physical dimension tensor.

        Returns:
            PhlowerDimensionTensor | None: Physical dimension tensor.
        """
        return self._dimension_tensor

    @property
    def shape(self) -> torch.Size:
        """Shape of the tensor.

        Returns:
            torch.Size: Shape of the tensor.
        """
        return self._phlower_shape.shape

    @property
    def shape_pattern(self) -> PhlowerShapePattern:
        """Shape pattern of the tensor.

        Returns:
            PhlowerShapePattern: Shape pattern of the tensor.
        """
        return self._phlower_shape

    @property
    def is_sparse(self) -> bool:
        """Whether the tensor is a sparse tensor.

        Returns:
            bool: Whether the tensor is a sparse tensor.
        """
        return self._tensor.layout == torch.sparse_coo

    @property
    def is_time_series(self) -> bool:
        """Whether the tensor is a time series.

        Returns:
            bool: Whether the tensor is a time series.
        """
        return self._phlower_shape.is_time_series

    @property
    def is_voxel(self) -> bool:
        """Whether the tensor is a voxel.

        Returns:
            bool: Whether the tensor is a voxel.
        """
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

    def __rsub__(self, other: PhlowerTensor):
        return torch.sub(other, self)

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

    def __matmul__(self, other: PhlowerTensor) -> PhlowerTensor:
        return torch.matmul(self, other)

    def __bool__(self) -> bool:
        return bool(self._tensor)

    def __array__(
        self, dtype: np.dtype | None = None, copy: bool | None = None
    ) -> np.ndarray:
        arr = self._tensor.numpy()
        return arr.astype(dtype) if dtype else arr

    def __getitem__(self, key: Any) -> PhlowerTensor:
        val = self._tensor[key]
        shape_pattern = self.shape_pattern.resolve_index_access(key, val.shape)
        return PhlowerTensor.from_pattern(
            val, dimension_tensor=self._dimension_tensor, pattern=shape_pattern
        )

    @functools.wraps(torch.Tensor.__setitem__)
    def __setitem__(self, key: Any, value: Any) -> Self:
        if isinstance(value, PhlowerTensor):
            if self.dimension != value.dimension:
                raise DimensionIncompatibleError(
                    "Dimension incompatible: "
                    f"{self.dimension} vs {value.dimension}"
                )
            value_to_set = value.to_tensor()
        else:
            value_to_set = value

        if isinstance(key, PhlowerTensor):
            self._tensor[key.to_tensor()] = value_to_set
        else:
            self._tensor[key] = value_to_set
        return self

    def __len__(self) -> int:
        return len(self._tensor)

    def to_phlower_array(self) -> IPhlowerArray:
        dimensions = (
            self.dimension.to_physics_dimension() if self.dimension else None
        )
        return phlower_array(
            data=self.to_numpy(),
            is_time_series=self.is_time_series,
            is_voxel=self.is_voxel,
            dimensions=dimensions,
        )

    def to_tensor(self) -> torch.Tensor:
        """Convert to torch.Tensor.

        Returns:
            torch.Tensor: Tensor.
        """
        return self._tensor

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy.ndarray.

        Returns:
            numpy.ndarray: Numpy array.
        """
        return self._tensor.cpu().detach().numpy()

    def numpy(self) -> np.ndarray:
        return self.to_numpy()

    def coalesce(self) -> torch.Tensor:
        """Coalesce the tensor.

        Returns:
            torch.Tensor: Coalesced tensor.
        """
        return PhlowerTensor(self._tensor.coalesce(), self._dimension_tensor)

    def size(self) -> torch.Size:
        """Size of the tensor.

        Returns:
            torch.Size: Size of the tensor.
        """
        return self._tensor.size()

    @property
    def dtype(self) -> torch.dtype:
        """Data type of the tensor.

        Returns:
            torch.dtype: Data type of the tensor.
        """
        return self._tensor.dtype

    @property
    def device(self) -> torch.device:
        """Device of the tensor.

        Returns:
            torch.device: Device of the tensor.
        """
        return self._tensor.device

    def transpose(self, dim0: int, dim1: int) -> PhlowerTensor:
        """Transpose the tensor.

        Returns:
            PhlowerTensor: Transposed tensor.
        """
        _tensor = self._tensor.transpose(dim0, dim1)
        return PhlowerTensor(
            tensor=_tensor,
            dimension_tensor=self._dimension_tensor,
            is_time_series=self.is_time_series,
            is_voxel=self.is_voxel,
        )

    def numel(self) -> int:
        """Number of elements in the tensor.

        Returns:
            int: Number of elements in the tensor.
        """
        return torch.numel(self._tensor)

    def rank(self) -> int:
        """Returns the tensor rank.

        Returns:
            int: Tensor rank.
        """
        if self.is_sparse:
            raise PhlowerSparseUnsupportedError(
                "Cannot call rank() for sparse PhlowerTensor"
            )

        return self._phlower_shape.rank_size

    def n_vertices(self) -> int:
        """
        Returns the number of vertices.

        Returns:
            int: Number of vertices.
        """
        if self.is_sparse:
            raise PhlowerSparseUnsupportedError(
                "Cannot call n_vertices() for sparse PhlowerTensor"
            )

        return self._phlower_shape.get_n_vertices()

    def is_global(self, n_batch: int) -> bool:
        """
        Returns True if the batched tensor is from global ones.

        Args:
            n_batch: int
                The number of batches.

        Returns:
            bool: True if the batched tensor is from global ones.
        """
        if self.is_sparse:
            return False
        return self._phlower_shape.is_global(n_batch)

    def indices(self) -> torch.Tensor:
        """
        Returns the indices of the tensor.

        Returns:
            torch.Tensor: Indices of the tensor.
        """
        return self._tensor.indices()

    def values(self) -> torch.Tensor:
        """
        Returns the values of the tensor.

        Returns:
            torch.Tensor: Values of the tensor.
        """
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
        """
        Rearrange the tensor.

        Returns:
            PhlowerTensor: Rearranged tensor.
        """
        rearranged = einops.rearrange(self._tensor, pattern, **axes_length)
        shape_pattern = self._phlower_shape.rearrange(pattern, rearranged.shape)

        return PhlowerTensor.from_pattern(
            rearranged, dimension_tensor=self.dimension, pattern=shape_pattern
        )

    def reshape(
        self,
        shape: Sequence[int],
        is_time_series: bool = False,
        is_voxel: bool = False,
    ) -> PhlowerTensor:
        """
        Reshape the tensor.

        Returns:
            PhlowerTensor: Reshaped tensor.
        """
        return PhlowerTensor(
            torch.reshape(self.to_tensor(), shape),
            dimension_tensor=self.dimension,
            is_time_series=is_time_series,
            is_voxel=is_voxel,
        )

    def slice_time(
        self,
        indices: int | slice | list[int] | np.ndarray | torch.Tensor,
        keep_time_series: bool = True,
    ) -> PhlowerTensor:
        """
        Slice the time series tensor.

        Returns:
            PhlowerTensor: Sliced tensor.
        """
        if not self.is_time_series:
            raise ValueError(
                "Not time series: \n"
                f"- shape: {self.shape}\n"
                f"- pattern: {self.shape_pattern}\n"
            )

        return self[[indices]] if keep_time_series else self[indices]

    def to(
        self,
        device: str | torch.device = None,
        non_blocking: bool = False,
        dtype: torch.dtype = None,
    ) -> PhlowerTensor:
        """
        Convert the tensor to a different device or data type.

        Returns:
            PhlowerTensor: Converted tensor.
        """
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
        """
        Detach the tensor.

        Returns:
            PhlowerTensor: Detached tensor.
        """
        if self.has_dimension:
            new_dimension = self._dimension_tensor.detach()
        else:
            new_dimension = None

        return PhlowerTensor(
            self._tensor.detach(),
            dimension_tensor=new_dimension,
            is_time_series=self.is_time_series,
            is_voxel=self.is_voxel,
        )

    def backward(self) -> None:
        """
        Backward the tensor.
        """
        self._tensor.backward()

    def as_pattern(self, pattern: str) -> PhlowerTensor:
        """
        Convert the tensor to a pattern.

        Returns:
            PhlowerTensor: Patterned tensor.
        """
        return PhlowerTensor.from_pattern(
            tensor=self._tensor,
            dimension_tensor=self._dimension_tensor,
            pattern=pattern,
        )

    def clone(self) -> PhlowerTensor:
        """
        Clone the tensor.

        Returns:
            PhlowerTensor: Cloned tensor.
        """
        if self.has_dimension:
            new_dimension = self._dimension_tensor.clone()
        else:
            new_dimension = None

        tensor = self._tensor.clone()
        return PhlowerTensor(
            tensor,
            dimension_tensor=new_dimension,
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
    ) -> PhlowerTensor | NamedTuple[IPhlowerTensor]:
        if func.__name__ in _UNSUPPORTED_FUNCTION_NAMES:
            raise PhlowerUnsupportedTorchFunctionError(
                f"Unsupported function: {func.__name__}."
                "Please use corresponding functional in "
                "`phlower.nn._functionals` modules."
            )

        kwargs = kwargs or {}

        _tensors = _recursive_resolve(args, "_tensor")

        # ret is tuple when torch.return_types.* is returned
        ret: torch.Tensor | tuple = func(*_tensors, **kwargs)

        tensor_args = [
            a
            for a in args
            if isinstance(a, PhlowerTensor)
            or (isinstance(a, list | tuple) and isinstance(a[0], PhlowerTensor))
        ]

        list_is_time_series = _recursive_resolve(tensor_args, "is_time_series")
        list_is_voxel = _recursive_resolve(tensor_args, "is_voxel")
        is_time_series = np.any(list_is_time_series).item()
        is_voxel = np.any(list_is_voxel).item()

        if _has_dimension(args):
            _dimensions = _recursive_resolve(
                args, "_dimension_tensor", allow_none=False
            )
            result_units = func(*_dimensions, **kwargs)
        else:
            # Unit calculation is not considered when unit tensor is not found.
            result_units = None

        if isinstance(ret, torch.Tensor):
            return PhlowerTensor(
                ret,
                result_units,
                is_time_series=is_time_series,
                is_voxel=is_voxel,
            )

        # Manage NamedTuple output from min or max function
        if isinstance(ret, torch.return_types.max | torch.return_types.min):
            extremum = PhlowerTensor(
                ret.values,
                result_units,
                is_time_series=is_time_series,
                is_voxel=is_voxel,
            )
            indices_unit = None if result_units is None else {}
            indices = phlower_tensor(
                ret.indices,
                indices_unit,
                is_time_series=is_time_series,
                is_voxel=is_voxel,
            )
            return ret.__class__((extremum, indices))

        raise PhlowerUnsupportedTorchFunctionError(
            f"return type {ret.__class__} is not supported yet."
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
            "Cannot calcualte PhlowerTensor with physics dimension "
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
