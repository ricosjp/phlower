from collections.abc import Sequence

import numpy as np
import torch

from phlower._base.tensors import PhlowerDimensionTensor, PhlowerTensor
from phlower._base.tensors._interface import IPhlowerTensor

from ._check import is_same_dimensions, is_same_layout


def concatenate(
    tensors: Sequence[IPhlowerTensor], dense_dim: int = 0
) -> IPhlowerTensor:
    if not is_same_layout(tensors):
        raise ValueError("Cannot concatenate dense tensor and sparse tensor")

    if not is_same_dimensions(tensors):
        raise ValueError(
            "Cannot concatenate tensors which have different dimensions."
        )

    _phy_dim = tensors[0].dimension
    if tensors[0].is_sparse:
        return _sparse_concatenate(tensors, _phy_dim)

    return _dense_concatenate(tensors, dim=dense_dim)


def _dense_concatenate(
    tensors: Sequence[IPhlowerTensor], dim: int = 0
) -> IPhlowerTensor:
    return torch.concatenate(tensors, dim=dim)


def _sparse_concatenate(
    tensors: Sequence[IPhlowerTensor], dimension: PhlowerDimensionTensor
) -> torch.Tensor:
    offsets = torch.cumsum(
        torch.tensor(
            [
                tensors[i - 1].shape if i != 0 else (0, 0)
                for i in range(len(tensors))
            ]
        ),
        axis=0,
    )

    shape = np.sum([arr.shape for arr in tensors], axis=0)

    rows = torch.concatenate(
        [arr.indices()[0] + offsets[i, 0] for i, arr in enumerate(tensors)],
        dim=0,
    )
    cols = torch.concatenate(
        [arr.indices()[1] + offsets[i, 1] for i, arr in enumerate(tensors)],
        dim=0,
    )

    data = torch.concatenate([arr.values() for arr in tensors], dim=0)

    sparse_tensor = torch.sparse_coo_tensor(
        indices=torch.stack([rows, cols]), values=data, size=tuple(shape)
    )
    return PhlowerTensor(sparse_tensor, dimension)
