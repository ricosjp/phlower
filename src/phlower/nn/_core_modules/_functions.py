
import torch

from phlower._base.tensors._interface import IPhlowerTensor


def spmm(
        sparse: IPhlowerTensor, x: IPhlowerTensor,
        repeat: int = 1) -> IPhlowerTensor:
    """
    Computes sparse matrix times dense tensor along with the vertex axis.

    Args:
        sparse : IPhlowerTensor:
            Sparse tensor.
        x : IPhlowerTensor
            Dense tensor.
        repeat : int, optional
            The number of repetitions for multiplication. The default is 1.
    Returns:
        IPhlowerTensor:
            Resultant tensor.
    """
    h, original_pattern, resultant_pattern, dict_shape = x.to_vertexwise()
    pattern = f"{resultant_pattern} -> {original_pattern}"
    for _ in range(repeat):
        h = torch.sparse.mm(sparse, h)
    return h.rearrange(
        pattern, is_time_series=x.is_time_series, is_voxel=x.is_voxel,
        **dict_shape)
