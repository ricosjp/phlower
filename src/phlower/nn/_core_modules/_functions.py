
import torch

from phlower._base._dimension import PhysicalDimensions
from phlower._base.tensors import phlower_tensor
from phlower._base.tensors._dimension_tensor import PhlowerDimensionTensor
from phlower._base.tensors._interface import IPhlowerTensor
from phlower.utils.exceptions import PhlowerIncompatibleTensorError


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


def contraction(
        x: IPhlowerTensor, y: IPhlowerTensor | None = None) -> IPhlowerTensor:
    """
    Compute the tensor contraction.

    Args:
        x : IPhlowerTensor
            Input tensor.
        y : IPhlowerTensor, optional
            Second input tensor. If not fed, it computes contraction of x.
    """
    if y is None:
        y = x

    if x.rank() < y.rank():
        return contraction(y, x)
    # Now we assume x.rank() >= y.rank()

    if x.is_voxel != y.is_voxel:
        raise PhlowerIncompatibleTensorError(
            "Cannot compute contraction between non-voxel and voxel.")

    ret_is_time_series = False
    if x.is_time_series:
        time_x = "t"
        ret_is_time_series = True
    else:
        time_x = ""
    if y.is_time_series:
        time_y = "t"
        ret_is_time_series = True
    else:
        time_y = ""
    if ret_is_time_series:
        time_ret = "t"
    else:
        time_ret = ""

    # No need to consider y because they should be compatible
    if x.is_voxel:
        space = "xyz"
        is_voxel = True
    else:
        space = "x"
        is_voxel = False

    diff_rank = x.rank() - y.rank()
    unresolved = "abcdeghijklmnopqrsuvw"[
        :diff_rank]  # No f, t, x, y, and z because they are used elsewhere

    if x.dimension is None or y.dimension is None:
        dimension = None
    else:
        dimension = x.dimension * y.dimension
    return einsum(
        f"{time_x}{space}...{unresolved}f,{time_y}{space}...f->"
        f"{time_ret}{space}{unresolved}f",
        x, y, dimension=dimension,
        is_time_series=ret_is_time_series, is_voxel=is_voxel)


def einsum(
        equation, *args: list[IPhlowerTensor],
        dimension: (
            PhysicalDimensions
            | PhlowerDimensionTensor
            | torch.Tensor
            | dict[str, float]
            | list[float]
            | tuple[float]
            | None
        ) = None,
        is_time_series: bool = False, is_voxel: bool = False,
) -> IPhlowerTensor:
    """
    Compute einsum for phlower tensors.

    Args:
        equation: str
            Equation for einsum operation.
        args: list[IPhlowerTensor]
            List of IPhlowerTensor objects.
        dimension:
                PhlowerDimensions | PhlowerDimensionTensor | torch.Tensor
                | dict[str, float] | list[float] | tuple[float] | None
            Dimension for the resultant tensor.
        is_time_series: bool, optional
            Flag for time series. The default is False.
        is_voxel: bool, optional
            Flag for voxel. The default is False.

    Returns:
        IPhlowerTensor:
            Resultant tensor
    """
    ret_tensor = torch.einsum(
        equation, [a.to_tensor() for a in args])
    return phlower_tensor(
        ret_tensor, dimension=dimension,
        is_time_series=is_time_series, is_voxel=is_voxel)
