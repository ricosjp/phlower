from collections.abc import Sequence

from pipe import select

from phlower._base._batch import GraphBatchInfo
from phlower._base.tensors._interface import IPhlowerTensor

from ._check import is_same_layout
from ._concatenate import concatenate


def to_batch(
    tensors: Sequence[IPhlowerTensor], dense_concat_dim: int = 0
) -> tuple[IPhlowerTensor, GraphBatchInfo]:
    if not is_same_layout(tensors):
        raise ValueError(
            "The same layout tensors can be converted to batched tensor."
            "Several layouts are found in an argument."
        )

    batch_info = _create_batch_info(tensors, dense_concat_dim)
    concat_tensor = concatenate(tensors, dense_dim=dense_concat_dim)

    return concat_tensor, batch_info


def _create_batch_info(
    tensors: Sequence[IPhlowerTensor], dense_concat_dim: int
) -> GraphBatchInfo:
    _shapes = list(tensors | select(lambda x: x.shape))
    _head = tensors[0]

    if _head.is_sparse:
        _sizes = list(tensors | select(lambda x: x.values().size()[0]))
        # NOTE: Assume dim=0 corresponds to n_nodes When sparse
        n_nodes = tuple(v[0] for v in _shapes)
    else:
        _sizes = list(tensors | select(lambda x: x.size()))
        n_nodes = tuple(v[dense_concat_dim] for v in _shapes)

    return GraphBatchInfo(
        _sizes, _shapes, n_nodes=n_nodes, dense_concat_dim=dense_concat_dim
    )
