import pytest
import torch
from phlower._base import PhlowerDimensionTensor, phlower_dimension_tensor


def generate_phlower_dimension_tensor(
    has_dimension: bool = True,
    inverse: bool = False,
) -> PhlowerDimensionTensor:
    if has_dimension:
        dimension = {"L": 2, "T": -1}
    else:
        dimension = {}
    if inverse:
        dimension = {k: -v for k, v in dimension.items()}
    return phlower_dimension_tensor(dimension)


def to_tensor_if_needed(
    x: PhlowerDimensionTensor | float,
) -> torch.Tensor | float:
    if isinstance(x, PhlowerDimensionTensor):
        return x.to_tensor()
    return x


def to_cuda_if_needed(
    x: PhlowerDimensionTensor | float,
) -> PhlowerDimensionTensor | float:
    if isinstance(x, PhlowerDimensionTensor):
        return x.to("cuda:0")
    return x


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div],
)
@pytest.mark.parametrize(
    "a, b",
    [
        (
            generate_phlower_dimension_tensor(),
            generate_phlower_dimension_tensor(),
        ),
    ],
)
def test__op_tensor_tensor_with_unit_on_gpu(
    op: callable,
    a: PhlowerDimensionTensor | float,
    b: PhlowerDimensionTensor | float,
):
    c = op(to_cuda_if_needed(a), to_cuda_if_needed(b))

    cpu_c = op(a, b)
    assert c == cpu_c.to("cuda:0")
    assert c.device == torch.device("cuda:0")


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "a, b",
    [
        (
            2.3,
            generate_phlower_dimension_tensor(),
        ),
        (
            generate_phlower_dimension_tensor(),
            2.3,
        ),
    ],
)
def test__mul_tensor_scalar_with_unit_on_gpu(
    a: PhlowerDimensionTensor | float,
    b: PhlowerDimensionTensor | float,
):
    c = torch.mul(to_cuda_if_needed(a), to_cuda_if_needed(b))

    assert c == generate_phlower_dimension_tensor().to("cuda:0")
    assert c.device == torch.device("cuda:0")


@pytest.mark.gpu_test
def test__div_tensor_scalar_with_unit_on_gpu():
    a = generate_phlower_dimension_tensor()
    b = 2.3
    c = torch.div(to_cuda_if_needed(a), to_cuda_if_needed(b))

    assert c == generate_phlower_dimension_tensor().to("cuda:0")
    assert c.device == torch.device("cuda:0")


@pytest.mark.gpu_test
def test__div_scalar_tensor_with_unit_on_gpu():
    a = 2.3
    b = generate_phlower_dimension_tensor()
    c = torch.div(to_cuda_if_needed(a), to_cuda_if_needed(b))

    assert c == generate_phlower_dimension_tensor(inverse=True).to("cuda:0")
    assert c.device == torch.device("cuda:0")


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div],
)
@pytest.mark.parametrize(
    "a, b",
    [
        (
            generate_phlower_dimension_tensor(has_dimension=False),
            generate_phlower_dimension_tensor(has_dimension=False),
        ),
        (
            2.3,
            generate_phlower_dimension_tensor(has_dimension=False),
        ),
        (
            generate_phlower_dimension_tensor(has_dimension=False),
            2.3,
        ),
    ],
)
def test__op_tensor_tensor_without_unit_on_gpu(
    op: callable,
    a: PhlowerDimensionTensor | float,
    b: PhlowerDimensionTensor | float,
):
    c = op(to_cuda_if_needed(a), to_cuda_if_needed(b))

    assert c.device == torch.device("cuda:0")
    assert c.is_dimensionless
