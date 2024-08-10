
import torch

from phlower._base._dimension import PhysicalDimensions
from phlower._base.tensors import phlower_tensor
from phlower._base.tensors._dimension_tensor import PhlowerDimensionTensor
from phlower._base.tensors._interface import IPhlowerTensor
from phlower.utils.exceptions import PhlowerIncompatibleTensorError


def identity(x):
    return x


def leaky_relu0p5(x):
    """Leaky ReLU with the negative slope = 0.5."""
    return torch.nn.functional.leaky_relu(x, negative_slope=0.5)


def inversed_leaky_relu0p5(x):
    """Inverse of leaky_relu0p5."""
    return torch.nn.functional.leaky_relu(x, negative_slope=2)


def truncated_atanh(x, epsilon=1e-8):
    """Inverse tanh with truncating values >=1 or <=-1."""
    x[x>=1. - epsilon] = 1. - epsilon
    x[x<=-1. + epsilon] = -1. + epsilon
    return torch.atanh(x)


def smooth_leaky_relu(x):
    """Smooth leaky ReLU"""
    # a x + (1 - a) x sqrt(x**2 + b)
    # return 0.75 * x + 0.25 * (x**2 + 1 / 16)**.5  # b = 1 / 16
    return 0.75 * x + 0.25 * (x**2 + 1 / 100)**.5  # b = 1 / 100


def inversed_smooth_leaky_relu(x):
    # return 1.5 * x - 1 / 16 * torch.sqrt((8 * x)**2 + 2)  # b = 1 / 16
    return 1.5 * x - 1 / 40 * torch.sqrt(400 * x**2 + 2)  # b = 1 / 100


def derivative_smooth_leaky_relu(x):
    # return 0.75 + 0.25 * x / torch.sqrt(x**2 + 1 / 16)  # b = 1 / 16
    return 0.75 + 0.25 * x / torch.sqrt(x**2 + 1 / 100)

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
    try:
        ret_tensor = torch.einsum(
            equation, [a.to_tensor() for a in args])
    except RuntimeError as e:
        raise PhlowerIncompatibleTensorError(
            f"{e}\n"
            f"{equation}, {[a.shape for a in args]}") from e
    return phlower_tensor(
        ret_tensor, dimension=dimension,
        is_time_series=is_time_series, is_voxel=is_voxel)


def _availale_variables(length: int, start: int = 0) -> str:
    # No f, t, x, y, and z because they are "reserved"
    available_variables = "abcdeghijklmnopqrsuvw"

    if length > len(available_variables):
        raise ValueError(f"Required length too long: {length}")
    return available_variables[start:start+length]


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
    unresolved = _availale_variables(diff_rank)

    if x.dimension is None or y.dimension is None:
        dimension = None
    else:
        dimension = x.dimension * y.dimension
    return einsum(
        f"{time_x}{space}...{unresolved}f,{time_y}{space}...f->"
        f"{time_ret}{space}{unresolved}f",
        x, y, dimension=dimension,
        is_time_series=ret_is_time_series, is_voxel=is_voxel)


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
            "Cannot compute contraction between non-voxel and voxel.")

    x_rank = x.rank()
    y_rank = y.rank()

    ret_is_time_series = False
    if x.is_time_series:
        x_time = "t"
        ret_is_time_series = True
    else:
        x_time = ""
    if y.is_time_series:
        y_time = "t"
        ret_is_time_series = True
    else:
        y_time = ""
    if ret_is_time_series:
        t_ret = "t"
    else:
        t_ret = ""

    # No need to consider y because they should be compatible
    if x.is_voxel:
        space = "xyz"
        is_voxel = True
    else:
        space = "x"
        is_voxel = False

    x_vars = _availale_variables(x_rank)
    y_vars = _availale_variables(y_rank, start=x_rank)
    equation = f"{x_time}{space}{x_vars}f,{y_time}{space}{y_vars}f->" \
        + f"{t_ret}{space}{x_vars}{y_vars}f"

    if x.dimension is None or y.dimension is None:
        dimension = None
    else:
        dimension = x.dimension * y.dimension

    return einsum(
        equation, x, y, dimension=dimension,
        is_time_series=ret_is_time_series, is_voxel=is_voxel)


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

    start_dim = 1
    if tensor.is_time_series:
        time = "t"
        start_dim += 1
    else:
        time = ""
    if tensor.is_voxel:
        space = "xyz"
        start_dim += 3
    else:
        space = "x"

    s = _availale_variables(rank * 2)
    str_ortho = ','.join(a + b for a, b in zip(s[::2], s[1::2], strict=True))
    str_tensor = f"{time}{space}{s[1::2]}f"
    str_ret = f"{time}{space}{s[::2]}f"
    equation = f"{str_ortho},{str_tensor}->{str_ret}"
    args = [orthogonal_matrix] * rank + [tensor]

    return einsum(
        equation, *args,
        dimension=tensor.dimension,
        is_time_series=tensor.is_time_series, is_voxel=tensor.is_voxel)
