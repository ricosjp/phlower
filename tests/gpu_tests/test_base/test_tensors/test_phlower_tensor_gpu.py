import numpy as np
import pytest
import torch
from phlower import PhlowerTensor, phlower_tensor


def generate_random_phlower_tensor(
    has_dimension: bool = True,
) -> PhlowerTensor:
    if has_dimension:
        dimension = {"L": 2, "T": -1}
    else:
        dimension = {}
    return phlower_tensor(torch.rand(5), dimension=dimension)


def to_tensor_if_needed(x: PhlowerTensor | float) -> torch.Tensor | float:
    if isinstance(x, PhlowerTensor):
        return x.to_tensor()
    return x


def to_cuda_if_needed(x: PhlowerTensor | float) -> PhlowerTensor | float:
    if isinstance(x, PhlowerTensor):
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
            generate_random_phlower_tensor(),
            generate_random_phlower_tensor(),
        ),
    ],
)
def test__op_tensor_tensor_with_unit_on_gpu(
    op: callable, a: PhlowerTensor | float, b: PhlowerTensor | float
):
    c = op(to_cuda_if_needed(a), to_cuda_if_needed(b))

    tc = op(to_tensor_if_needed(a), to_tensor_if_needed(b))
    np.testing.assert_almost_equal(c.numpy(), tc.cpu().numpy())
    assert c.device == torch.device("cuda:0")
    assert c.device == torch.device("cuda:0")


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "op",
    [torch.mul, torch.div],
)
@pytest.mark.parametrize(
    "a, b",
    [
        (
            2.3,
            generate_random_phlower_tensor(),
        ),
        (
            generate_random_phlower_tensor(),
            2.3,
        ),
    ],
)
def test__op_tensor_scalar_with_unit_on_gpu(
    op: callable, a: PhlowerTensor | float, b: PhlowerTensor | float
):
    c = op(to_cuda_if_needed(a), to_cuda_if_needed(b))

    tc = op(to_tensor_if_needed(a), to_tensor_if_needed(b))
    np.testing.assert_almost_equal(c.numpy(), tc.cpu().numpy())
    assert c.device == torch.device("cuda:0")
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
            generate_random_phlower_tensor(has_dimension=False),
            generate_random_phlower_tensor(has_dimension=False),
        ),
        (
            2.3,
            generate_random_phlower_tensor(has_dimension=False),
        ),
        (
            generate_random_phlower_tensor(has_dimension=False),
            2.3,
        ),
    ],
)
def test__op_tensor_tensor_without_unit_on_gpu(
    op: callable, a: PhlowerTensor | float, b: PhlowerTensor | float
):
    c = op(to_cuda_if_needed(a), to_cuda_if_needed(b))

    tc = op(to_tensor_if_needed(a), to_tensor_if_needed(b))
    np.testing.assert_almost_equal(c.numpy(), tc.cpu().numpy())
    assert c.device == torch.device("cuda:0")
    assert c.dimension.device == torch.device("cuda:0")
