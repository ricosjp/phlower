import torch

from phlower._base import phlower_tensor
from phlower._base._batch import GraphBatchInfo
from phlower._base.tensors._interface import IPhlowerTensor


def unbatch(
    tensor: IPhlowerTensor, batch_info: GraphBatchInfo
) -> list[IPhlowerTensor]:
    if tensor.is_sparse:
        results = _sparse_unbatch(tensor.to_tensor(), batch_info)
    else:
        results = _dense_unbatch(tensor.to_tensor(), batch_info)

    return [phlower_tensor(v, tensor.dimension) for v in results]


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
    tensor: torch.Tensor, batch_info: GraphBatchInfo
) -> list[torch.Tensor]:
    n_nodes = batch_info.total_n_nodes

    index_dim = _get_n_node_dim(tensor, n_nodes)
    n_rows = [v[index_dim] for v in batch_info.shapes]
    return tensor.split(n_rows, dim=index_dim)


def _get_n_node_dim(tensor: torch.Tensor, n_nodes: int) -> int:
    _index = [i for i, v in enumerate(tensor.shape) if v == n_nodes]

    if len(_index) == 0:
        raise ValueError(
            f"concatenated dimension is not found. shape: {tensor.shape}"
            f", n_nodes: {n_nodes}"
        )

    if len(_index) > 1:
        raise ValueError(
            f"multiple concatenated dimensions are found. shape: {tensor.shape}"
            f", n_nodes: {n_nodes}"
        )

    return _index[0]
