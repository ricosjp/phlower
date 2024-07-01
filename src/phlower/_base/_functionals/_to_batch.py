from collections.abc import Sequence

from pipe import select

from phlower._base._batch import BatchInfo
from phlower._base.tensors._interface import IPhlowerTensor

from ._check import is_same_layout
from ._concatenate import concatenate


def to_batch(
    tensors: Sequence[IPhlowerTensor], dense_concat_dim: int = 0
) -> tuple[IPhlowerTensor, BatchInfo]:
    if not is_same_layout(tensors):
        raise ValueError(
            "The same layout tensors can be converted to batched tensor."
            "Several layouts are found in an argument."
        )

    batch_info = _create_batch_info(tensors)
    concat_tensor = concatenate(tensors, dense_dim=dense_concat_dim)

    return concat_tensor, batch_info


def _create_batch_info(tensors: Sequence[IPhlowerTensor]):
    _shapes = list(tensors | select(lambda x: x.shape))

    if tensors[0].is_sparse:
        _sizes = list(tensors | select(lambda x: x.values().size()[0]))
    else:
        _sizes = list(tensors | select(lambda x: x.size()))

    return BatchInfo(_sizes, _shapes)
