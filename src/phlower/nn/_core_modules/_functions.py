
import torch

from phlower._base.tensors._interface import IPhlowerTensor


def spmm(sparse: IPhlowerTensor, x: IPhlowerTensor) -> IPhlowerTensor:
    h = x.to_2d()
    ret = torch.sparse.mm(sparse, h)
    pattern = f"{h.current_pattern} -> {h.original_pattern}"
    return ret.rearrange(
        pattern, is_time_series=x.is_time_series, is_voxel=x.is_voxel,
        **h.dict_shape)
