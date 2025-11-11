from typing import NamedTuple

import einops
import torch

from phlower._base.tensors import PhlowerDimensionTensor, phlower_tensor
from phlower._base.tensors._interface import IPhlowerTensor
from phlower._base.tensors._phlower_tensor import PhysicDimensionLikeObject
from phlower.utils.exceptions import PhlowerIncompatibleTensorError


def spmm(
    sparse: IPhlowerTensor, x: IPhlowerTensor, repeat: int = 1
) -> IPhlowerTensor:
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
    h, resultant_pattern = x.to_vertexwise()
    restore_pattern = f"{resultant_pattern} -> {x.shape_pattern.get_pattern()}"
    restore_axes_length = x.shape_pattern.get_pattern_to_size(drop_last=True)
    if (
        symbol := x.shape_pattern.n_nodes_pattern_symbol
    ) in restore_axes_length:
        restore_axes_length[symbol] = sparse.shape[0]

    for _ in range(repeat):
        h = torch.sparse.mm(sparse, h)
    return h.rearrange(restore_pattern, **restore_axes_length)


def time_series_to_features(arg: IPhlowerTensor) -> IPhlowerTensor:
    """
    Compute time_series_to_features for phlower tensors.

    Args:
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

    if not arg.is_time_series:
        raise ValueError(
            "input tensor is not timeseries tensor. "
            "time_series_to_features cannot be computed."
        )

    original_pattern = arg.shape_pattern.get_pattern()
    patterns = original_pattern.split(" ")
    resultant_pattern = " ".join(
        patterns[1:-1] + [f"({patterns[-1]} {patterns[0]})"]
    )

    _to_tensor = einops.rearrange(
        arg.to_tensor(), f"{original_pattern} -> {resultant_pattern}"
    )

    return phlower_tensor(
        _to_tensor, dimension=arg.dimension, pattern=resultant_pattern
    )


class BNFRestoreInfo(NamedTuple):
    """Contains information for restoring original shape pattern
    from batch-node-feature (BNF) format.
    """

    original_pattern: str
    bnf_pattern: str
    restore_axes_length: str
    dimension: PhlowerDimensionTensor | None

    @property
    def restore_pattern(self) -> str:
        return f"{self.bnf_pattern} -> {self.original_pattern}"


def to_batch_node_feature(
    arg: IPhlowerTensor,
) -> tuple[torch.Tensor, BNFRestoreInfo]:
    """Convert tensor's shape to batch-node-feature (BNF) format.

    Collapses time and rank dimensions into a single batch dimension,
    resulting in a 3D tensor: (batch, nodes, features).

    Args:
        arg: IPhlowerTensor
            Input tensor.

    Returns:
        resultant_tensor: torch.Tensor
            Resultant tensor.
        restore_info: BNFRestoreInfo
            Information for restoring original shape.

    Examples:
        >>> # Input tensor of shape (5, 100, 2, 2, 3) with pattern 't n d0 d1 f'
        >>> result, pattern, axes = to_batch_node_feature(tensor)
        >>> result.shape  # (20, 100, 3) with pattern '(t d0 d1) n f'
    """
    original_pattern = arg.shape_pattern.get_pattern()
    restore_axes_length = arg.shape_pattern.get_pattern_to_size()

    time_pattern = arg.shape_pattern.time_series_pattern
    space_pattern = arg.shape_pattern.get_space_pattern()
    ranks_pattern = arg.shape_pattern.get_feature_pattern(drop_last=True)
    feature_pattern = arg.shape_pattern.feature_pattern_symbol

    # The feature dimension can change when restoring
    restore_axes_length.pop(feature_pattern)

    # Combine time and ranks into batch dimension
    batch_components = [x for x in [time_pattern, ranks_pattern] if x]
    batch_pattern = " ".join(batch_components) if batch_components else "1"

    bnf_pattern = f"({batch_pattern}) ({space_pattern}) ({feature_pattern})"
    pattern = f"{original_pattern} -> {bnf_pattern}"
    _to_tensor = einops.rearrange(arg.to_tensor(), pattern)

    return (
        _to_tensor,
        BNFRestoreInfo(
            original_pattern,
            bnf_pattern,
            restore_axes_length,
            arg.dimension,
        ),
    )


def from_batch_node_feature(
    arg: torch.Tensor, restore_info: BNFRestoreInfo
) -> IPhlowerTensor:
    """Convert back tensor's shape from batch-node-feature (BNF)
    to original format.
    To use after `to_batch_node_feature` to get back original shape.

    Args:
        arg: torch.Tensor
            Input tensor.
        restore_info: BNFRestoreInfo
            Information for restoring original pattern.

    Returns:
        resultant_tensor: IPhlowerTensor
            Resultant tensor.
    """
    original_pattern = restore_info.original_pattern
    pattern = restore_info.restore_pattern
    axes_length = restore_info.restore_axes_length
    dimension = restore_info.dimension

    _to_tensor = einops.rearrange(arg, pattern, **axes_length)

    return phlower_tensor(
        _to_tensor, dimension=dimension, pattern=original_pattern
    )


def einsum(
    equation: str,
    *args: IPhlowerTensor,
    dimension: PhysicDimensionLikeObject | None = None,
    is_time_series: bool | None = None,
    is_voxel: bool | None = None,
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
    try:
        ret_tensor = torch.einsum(equation, [a.to_tensor() for a in args])
    except RuntimeError as e:
        raise PhlowerIncompatibleTensorError(
            f"{e}\n" f"{equation}, {[a.shape for a in args]}"
        ) from e

    is_none_time = is_time_series is None
    is_none_voxel = is_voxel is None

    if is_none_time and is_none_voxel:
        pattern = equation.split("->")[-1]
        return phlower_tensor(ret_tensor, dimension=dimension, pattern=pattern)

    return phlower_tensor(
        ret_tensor,
        dimension=dimension,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
    )


def _availale_variables(length: int, start: int = 0) -> str:
    # No f, t, x, y, n and z because they are "reserved"
    available_variables = "abcdeghijklmopqrsuvw"

    if length > len(available_variables):
        raise ValueError(f"Required length too long: {length}")
    return available_variables[start : start + length]


def contraction(
    x: IPhlowerTensor, y: IPhlowerTensor | None = None
) -> IPhlowerTensor:
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
            "Cannot compute contraction between non-voxel and voxel."
        )

    ret_is_time_series = x.is_time_series or y.is_time_series
    time_x = x.shape_pattern.time_series_pattern
    time_y = y.shape_pattern.time_series_pattern
    time_ret = "t" if ret_is_time_series else ""

    # No need to consider y because they should be compatible
    space = x.shape_pattern.get_space_pattern(omit_space=True)

    diff_rank = x.rank() - y.rank()
    unresolved = _availale_variables(diff_rank)

    if x.dimension is None or y.dimension is None:
        dimension = None
    else:
        dimension = x.dimension * y.dimension
    return einsum(
        f"{time_x}{space}...{unresolved}f,{time_y}{space}...f->"
        f"{time_ret}{space}{unresolved}f",
        x,
        y,
        dimension=dimension,
    )


def inner_product(
    x: IPhlowerTensor, y: IPhlowerTensor | None = None
) -> IPhlowerTensor:
    """
    Compute the inner products with interactions.

    Args:
        x : IPhlowerTensor
            (..., f1)-shaped input tensor.
        y : IPhlowerTensor, optional
            (..., f2)-shaped second input tensor.
            If not fed, it computes inner products of x.

    Returns:
        IPhlowerTensor
            (..., f1 * f2)-shaped tensor
    """
    if y is None:
        y = x

    if x.rank() != y.rank():
        raise ValueError(
            "Rank should be the same for inner product: "
            f"{x.rank()} vs {y.rank()}"
        )

    if x.is_voxel != y.is_voxel:
        raise PhlowerIncompatibleTensorError(
            "Cannot compute inner products between non-voxel and voxel."
        )

    ret_is_time_series = x.is_time_series or y.is_time_series
    time_x = x.shape_pattern.time_series_pattern
    time_y = y.shape_pattern.time_series_pattern
    time_ret = "t" if ret_is_time_series else ""

    # No need to consider y because they should be compatible
    space = x.shape_pattern.get_space_pattern(omit_space=True)

    if x.dimension is None or y.dimension is None:
        dimension = None
    else:
        dimension = x.dimension * y.dimension

    return einsum(
        f"{time_x}{space}...f,{time_y}{space}...g->{time_ret}{space}fg",
        x,
        y,
        dimension=dimension,
    ).rearrange("... f g -> ... (f g)")


def tensor_product(x: IPhlowerTensor, y: IPhlowerTensor) -> IPhlowerTensor:
    """
    Compute spatial tensor product for input tensors.

    Args:
        x: IPhlowerTensor
        y: IPhlowerTensor

    Returns:
        IPhlowerTensor:
            Resultant tensor.
    """
    if x.is_voxel != y.is_voxel:
        raise PhlowerIncompatibleTensorError(
            "Cannot compute contraction between non-voxel and voxel."
        )

    x_rank = x.rank()
    y_rank = y.rank()

    ret_is_time_series = x.is_time_series or y.is_time_series
    time_x = x.shape_pattern.time_series_pattern
    time_y = y.shape_pattern.time_series_pattern
    time_ret = "t" if ret_is_time_series else ""

    # No need to consider y because they should be compatible
    space = x.shape_pattern.get_space_pattern(omit_space=True)

    x_vars = _availale_variables(x_rank)
    y_vars = _availale_variables(y_rank, start=x_rank)
    equation = (
        f"{time_x}{space}{x_vars}f,{time_y}{space}{y_vars}f->"
        + f"{time_ret}{space}{x_vars}{y_vars}f"
    )

    if x.dimension is None or y.dimension is None:
        dimension = None
    else:
        dimension = x.dimension * y.dimension

    return einsum(equation, x, y, dimension=dimension)


def tensor_times_scalar(
    tensor: IPhlowerTensor, scalar: int | float | IPhlowerTensor
) -> IPhlowerTensor:
    """
    Compute multiplication between tensor and scalar (field).

    Args:
        tensor: IPhlowerTensor
            Input tensor.
        scalar: int | float | IPhlowerTensor
            Input scalar or scalar field.

    Returns:
        IPhlowerTensor:
            Resultant tensor.
    """
    if isinstance(scalar, int | float):
        return tensor * scalar
    if scalar.numel() == 1:
        return tensor * scalar

    return tensor_product(tensor, scalar)


def apply_orthogonal_group(
    orthogonal_matrix: IPhlowerTensor, tensor: IPhlowerTensor
) -> IPhlowerTensor:
    """
    Apply orthogonal group action to the input tensor.

    Args:
        orthogonal_matrix: IPhlowerTensor
            [3, 3]-shaped orthogonal matrix.
        tensor: IPhlowerTensor
            Tensor to apply the orthogonal group action.

    Returns:
        IPhlowerTensor:
            Resultant tensor.
    """
    rank = tensor.rank()
    if rank == 0:
        return tensor

    time = tensor.shape_pattern.time_series_pattern
    space = tensor.shape_pattern.get_space_pattern(omit_space=True)

    s = _availale_variables(rank * 2)
    str_ortho = ",".join(a + b for a, b in zip(s[::2], s[1::2], strict=True))
    str_tensor = f"{time}{space}{s[1::2]}f"
    str_ret = f"{time}{space}{s[::2]}f"
    equation = f"{str_ortho},{str_tensor}->{str_ret}"
    args = [orthogonal_matrix] * rank + [tensor]

    return einsum(equation, *args, dimension=tensor.dimension)


def spatial_sum(tensor: IPhlowerTensor) -> IPhlowerTensor:
    """Compute sum over space."""

    if len(tensor.shape) == 1 and tensor.numel() == 1:
        return tensor

    time = tensor.shape_pattern.time_series_pattern
    space = tensor.shape_pattern.get_space_pattern(omit_space=True)
    start_space = tensor.shape_pattern.start_space_index
    space_width = tensor.shape_pattern.space_width

    squeezed = einsum(
        f"{time}{space}...->{time}...", tensor, dimension=tensor.dimension
    )

    # keepdim
    for _ in range(space_width):
        squeezed = torch.unsqueeze(squeezed, start_space)

    return squeezed.as_pattern(f"{time}{space}...")


def spatial_mean(
    tensor: IPhlowerTensor, weight: IPhlowerTensor = None
) -> IPhlowerTensor:
    """Compute mean over space."""
    if weight is None:
        return spatial_sum(tensor) / tensor.n_vertices()

    # Weighted mean
    return spatial_sum(tensor_times_scalar(tensor, weight)) / spatial_sum(
        weight
    )


def squeeze(input: IPhlowerTensor, dim: int | None = None) -> IPhlowerTensor:
    """
    Squeeze the input tensor along the specified dimension.

    Args:
        input: IPhlowerTensor
            Input tensor.
        dim: int | None, optional
            Dimension to squeeze. If None, squeezes all dimensions of size 1.
            The default is None.

    Returns:
        IPhlowerTensor:
            Squeezed tensor.
    """

    kwards = {"dim": dim} if dim is not None else {}
    val = torch.squeeze(input.to_tensor(), **kwards)
    pattern = input.shape_pattern.squeeze(dim=dim)

    return phlower_tensor(
        val,
        dimension=input.dimension,
        pattern=pattern,
    )
