from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Iterable

import numpy as np
import torch

from phlower._base._batch import SparseBatchInfo
from phlower._base.tensors._dimensions import (
    PhlowerDimensionTensor,
    phlower_dimension_tensor,
)
from phlower.utils import get_logger

logger = get_logger(__name__)


def phlower_tensor(
    tensor: torch.Tensor | PhlowerTensor,
    dimension: (
        PhlowerDimensionTensor | torch.Tensor | dict[str, float] | None
    ) = None,
    sparse_batch_info: SparseBatchInfo | None = None,
):
    if isinstance(tensor, PhlowerTensor):
        if (dimension is not None) or (sparse_batch_info is not None):
            logger.warning(
                "Input dimension_tensor and sparse_batch_info are ignored."
            )

        return tensor

    if isinstance(tensor, torch.Tensor):
        dimension_tensor = _resolve_dimension_arg(dimension)

        return PhlowerTensor(
            tensor=tensor,
            dimension_tensor=dimension_tensor,
            sparse_batch_info=sparse_batch_info,
        )


def _resolve_dimension_arg(
    inputs: PhlowerDimensionTensor | torch.Tensor | dict[str, float] | None,
) -> PhlowerDimensionTensor | None:
    if inputs is None:
        return None

    if isinstance(inputs, PhlowerDimensionTensor):
        return inputs

    if isinstance(inputs, torch.Tensor):
        return PhlowerDimensionTensor(inputs)

    if isinstance(inputs, dict):
        return phlower_dimension_tensor(inputs)

    raise NotImplementedError(
        f"{type(inputs)} is not implemented "
        "when creating PhlowerDimensionTensor"
    )


class PhlowerTensor:

    def __init__(
        self,
        tensor: torch.Tensor,
        dimension_tensor: PhlowerDimensionTensor | None = None,
        sparse_batch_info: SparseBatchInfo | None = None,
    ):
        assert isinstance(tensor, torch.Tensor)
        self._tensor = tensor
        self._dimension_tensor = dimension_tensor
        self._sparse_batch_info = sparse_batch_info

    @property
    def has_dimension(self) -> bool:
        return self._dimension_tensor is not None

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    def __str__(self) -> str:
        return (
            f"PhysicsTensor({self._tensor}, "
            f"Dimension: {self._dimension_tensor})"
        )

    def __add__(self, other) -> PhlowerTensor:
        return torch.add(self, other)

    def __mul__(self, other) -> PhlowerTensor:
        return torch.mul(self, other)

    def __len__(self) -> int:
        return len(self._tensor)

    def __getitem__(self, key: Any) -> PhlowerTensor:
        return PhlowerTensor(self._tensor[key], self._dimension_tensor)

    def tensor(self) -> torch.Tensor:
        return self._tensor

    def size(self) -> torch.Size:
        return self._tensor.size()

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

    def decompose(self) -> list[torch.Tensor]:
        if self._sparse_batch_info is None:
            return [self._tensor]

        return _sparse_decompose(self._tensor, self._sparse_batch_info)

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
            # HACK: NEED TO PASS self._shapes ??
            return PhlowerTensor(ret)

        _dimensions = _recursive_resolve(args, "_dimension_tensor")
        result_units = func(*_dimensions, **kwargs)
        # HACK: NEED TO PASS self._shapes ??
        return PhlowerTensor(ret, result_units)


def _recursive_resolve(args: Iterable | Any, attr: str = None) -> list[str]:
    if isinstance(args, (tuple, list)):
        return [_recursive_resolve(v, attr) for v in args]

    return getattr(args, attr, args)


def _has_dimension(args: Any) -> bool:
    if isinstance(args, (tuple, list)):
        return any([_has_dimension(v) for v in args])

    if isinstance(args, dict):
        return _has_dimension(list(args.values()))

    if isinstance(args, PhlowerTensor):
        return args.has_dimension

    return False


def _sparse_decompose(tensor: PhlowerTensor, batch_info: SparseBatchInfo):
    sizes = np.cumsum(batch_info.sizes)
    offsets = np.cumsum(
        np.array([[0, 0]] + batch_info.shapes[:-1], dtype=np.int32), axis=0
    )

    sparse_tensor = tensor.tensor()

    rows = np.split(sparse_tensor.indices[0], sizes)
    cols = np.split(sparse_tensor.indices[1], sizes)
    data = np.split(sparse_tensor.values, sizes)

    results = [
        torch.sparse_coo_tensor(
            indices=[rows[i] - offsets[i][0], cols[i] - offsets[i][1]],
            values=data[i],
            size=batch_info.shapes[i],
        )
        for i in range(len(batch_info))
    ]
    return results
