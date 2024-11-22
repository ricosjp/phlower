import torch

from phlower._base import PhlowerTensor
from phlower.collections import IPhlowerTensorCollections


def calculate_residual(
    x: IPhlowerTensorCollections,
    ref: IPhlowerTensorCollections,
    keys: list[str] | None = None,
) -> PhlowerTensor:
    if keys is None:
        keys = list(x.keys())

    return torch.sum(
        torch.stack([torch.linalg.norm(x[k], ref[k]) for k in keys])
    )
