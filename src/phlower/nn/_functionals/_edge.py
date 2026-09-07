from __future__ import annotations

import torch
from phlower_tensor import PhlowerTensor


def extract_edge_indices(
    support: PhlowerTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the directed edge list from a sparse support tensor.

    Each nonzero entry (i, j) of the support corresponds
    to an edge of receiver i and sender j.
    Self-loop edges are removed.

    Args:
        support: PhlowerTensor
            sparse tensor of shape (n_nodes, n_nodes)

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            receiver and sender node indices, each of shape (n_edges,)
    """
    indices = support.coalesce().indices()  # (2, nnz)
    mask = indices[0] != indices[1]
    return indices[0][mask], indices[1][mask]  # (receiver, sender)
