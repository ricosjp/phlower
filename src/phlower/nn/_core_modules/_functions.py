import torch

from phlower._base.tensors import phlower_tensor
from phlower._base.tensors._interface import IPhlowerTensor
from phlower._base.tensors._phlower_tensor import PhysicDimensionLikeObject
from phlower.utils.exceptions import PhlowerIncompatibleTensorError


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def leaky_relu0p5(x: torch.Tensor) -> torch.Tensor:
    """Leaky ReLU with the negative slope = 0.5."""
    return torch.nn.functional.leaky_relu(x, negative_slope=0.5)


def inversed_leaky_relu0p5(x: torch.Tensor) -> torch.Tensor:
    """Inverse of leaky_relu0p5."""
    return torch.nn.functional.leaky_relu(x, negative_slope=2)


def truncated_atanh(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Inverse tanh with truncating values >=1 or <=-1."""
    x[x >= 1.0 - epsilon] = 1.0 - epsilon
    x[x <= -1.0 + epsilon] = -1.0 + epsilon
    return torch.atanh(x)


class SmoothLeakyReLU:
    """Smooth leaky ReLU"""

    def __init__(self, a: float = 0.75, b: float = 1 / 100):
        self.a = a
        self.b = b

    def __call__(self, x: torch.Tensor):
        # a x + (1 - a) sqrt(x**2 + b)
        return self.a * x + (1 - self.a) * torch.sqrt(x**2 + self.b)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.a * x
            - torch.sqrt(
                (self.a - 1) ** 2 * (2 * self.a * self.b - self.b + x**2)
            )
        ) / (2 * self.a - 1)

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return self.a - ((self.a - 1) * x) / torch.sqrt(self.b + x**2)


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

    for _ in range(repeat):
        h = torch.sparse.mm(sparse, h)
    return h.rearrange(restore_pattern, **restore_axes_length)


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
    space = x.shape_pattern.space_pattern

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
    space = x.shape_pattern.space_pattern

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
    space = tensor.shape_pattern.space_pattern

    s = _availale_variables(rank * 2)
    str_ortho = ",".join(a + b for a, b in zip(s[::2], s[1::2], strict=True))
    str_tensor = f"{time}{space}{s[1::2]}f"
    str_ret = f"{time}{space}{s[::2]}f"
    equation = f"{str_ortho},{str_tensor}->{str_ret}"
    args = [orthogonal_matrix] * rank + [tensor]

    return einsum(equation, *args, dimension=tensor.dimension)


def spatial_sum(tensor: IPhlowerTensor) -> IPhlowerTensor:
    """Compute sum over space."""

    time = tensor.shape_pattern.time_series_pattern
    space = tensor.shape_pattern.space_pattern
    start_space = 1 if tensor.is_time_series else 0
    space_width = 3 if tensor.is_voxel else 1

    squeezed = einsum(
        f"{time}{space}...->{time}...", tensor, dimension=tensor.dimension
    )

    # keepdim
    for _ in range(space_width):
        squeezed = torch.unsqueeze(squeezed, start_space)

    return squeezed.as_pattern(f"{time}{space}...")


def spatial_mean(tensor: IPhlowerTensor) -> IPhlowerTensor:
    """Compute mean over space."""
    return spatial_sum(tensor) / tensor.n_vertices()
