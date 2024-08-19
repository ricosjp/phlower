from collections.abc import Sequence

from pipe import select, uniq

from phlower._base.tensors import PhlowerDimensionTensor
from phlower._base.tensors._interface import IPhlowerTensor


def is_same_layout(tensors: Sequence[IPhlowerTensor]) -> bool:
    is_sparse_flags: set[bool] = set(tensors | select(lambda x: x.is_sparse))

    return len(is_sparse_flags) == 1


def is_same_dimensions(tensors: Sequence[IPhlowerTensor]) -> bool:
    if len(tensors) == 0:
        return True

    dimensions: set[PhlowerDimensionTensor] = list(
        tensors | select(lambda x: x.dimension == tensors[0].dimension) | uniq
    )

    return len(dimensions) == 1
