from typing import overload

import torch

from phlower._base import phlower_tensor
from phlower._base._batch import GraphBatchInfo
from phlower._base.tensors._interface import IPhlowerTensor


@overload
def unbatch(
    tensor: IPhlowerTensor, n_nodes: list[int]
) -> list[IPhlowerTensor]: ...


@overload
def unbatch(
    tensor: IPhlowerTensor, batch_info: GraphBatchInfo
) -> list[IPhlowerTensor]: ...


def unbatch(
    tensor: IPhlowerTensor,
    batch_info: GraphBatchInfo | None = None,
    n_nodes: list[int] | None = None,
) -> list[IPhlowerTensor]:
    if (not batch_info) and (not n_nodes):
        raise ValueError(
            "Parameters are missing. batch_info or n_nodes is necessary."
        )

    if tensor.is_sparse:
        if batch_info is None:
            raise ValueError(
                "batch_info is necessary when unbatching sparse tensor."
            )
        results = _sparse_unbatch(tensor.to_tensor(), batch_info)
        return [phlower_tensor(v, tensor.dimension) for v in results]

    # Only for dense tensor

    if batch_info:
        n_nodes = batch_info.n_nodes
        if not batch_info.is_concatenated:
            return [tensor]

    if tensor.is_voxel:
        if batch_info is None:
            raise ValueError(
                "batch_info is necessary when unbatching voxel tensor."
            )
        results = _dense_unbatch(
            tensor.to_tensor(), n_nodes, batch_info.dense_concat_dim
        )
    else:
        results = _dense_unbatch(
            tensor.to_tensor(), n_nodes, tensor.shape_pattern.nodes_dim
        )

    return [
        phlower_tensor(
            v, dimension=tensor.dimension, is_time_series=tensor.is_time_series
        )
        for v in results
    ]


def _sparse_unbatch(
    sparse_tensor: torch.Tensor, batch_info: GraphBatchInfo
) -> list[torch.Tensor]:
    sizes = torch.tensor(batch_info.sizes, dtype=torch.int32)
    offsets = torch.tensor(
        torch.cumsum(
            torch.tensor([[0, 0]] + batch_info.shapes[:-1], dtype=torch.int32),
            axis=0,
            dtype=torch.int32,
        )
    )
    sparse_tensor = sparse_tensor.coalesce()

    rows = sparse_tensor.indices()[0] - offsets[:, 0].repeat_interleave(sizes)
    rows = rows.split(batch_info.sizes)

    cols = sparse_tensor.indices()[1] - offsets[:, 1].repeat_interleave(sizes)
    cols = cols.split(batch_info.sizes)

    data = sparse_tensor.values().split(batch_info.sizes)

    results = [
        torch.sparse_coo_tensor(
            indices=torch.stack([rows[i], cols[i]]),
            values=data[i],
            size=batch_info.shapes[i],
        )
        for i in range(len(batch_info))
    ]
    return results


def _dense_unbatch(
    tensor: torch.Tensor, n_nodes: tuple[int], node_dim: int
) -> list[torch.Tensor]:
    return tensor.split(n_nodes, dim=node_dim)
