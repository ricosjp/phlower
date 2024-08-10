import numpy as np
import pytest
import torch

from phlower import PhlowerTensor, phlower_dimension_tensor, phlower_tensor
from phlower.utils.exceptions import (
    DimensionIncompatibleError,
    PhlowerSparseUnsupportedError,
)


def test__add():
    a = torch.eye(5)
    b = torch.eye(5)
    c = (a + b).numpy()

    ap = PhlowerTensor(a)
    bp = PhlowerTensor(b)
    cp = ap + bp
    cp = cp.to_tensor().numpy()

    np.testing.assert_array_almost_equal(cp, c)


def test__add_with_unit():
    units = phlower_dimension_tensor({"L": 2, "T": -2})
    a = np.random.rand(3, 10)
    b = np.random.rand(3, 10)
    c = a + b

    ap = PhlowerTensor(torch.tensor(a), units)
    bp = PhlowerTensor(torch.tensor(b), units)
    cp = ap + bp

    np.testing.assert_array_almost_equal(cp.to_tensor().numpy(), c)


@pytest.mark.parametrize(
    "unit1, unit2", [({"L": 2, "T": -2}, None), (None, {"M": 2, "T": -3})]
)
def test__add_with_and_without_dimensions(unit1, unit2):
    tensor1 = phlower_tensor(torch.rand(3, 4), dimension=unit1)
    tensor2 = phlower_tensor(torch.rand(3, 4), dimension=unit2)

    with pytest.raises(DimensionIncompatibleError):
        _ = tensor1 + tensor2


def test__add_with_unit_incompatible():
    units_1 = phlower_dimension_tensor({"L": 2, "T": -2})
    units_2 = phlower_dimension_tensor({"M": 1, "T": -2})

    a = torch.from_numpy(np.random.rand(3, 10))
    b = torch.from_numpy(np.random.rand(3, 10))
    with pytest.raises(DimensionIncompatibleError):
        ap = PhlowerTensor(a, units_1)
        bp = PhlowerTensor(b, units_2)
        _ = ap + bp


def test__mul_with_unit():
    dims_1 = phlower_dimension_tensor({"L": 2, "T": -2})
    dims_2 = phlower_dimension_tensor({"M": 1, "T": -2})
    dims_3 = phlower_dimension_tensor({"L": 2, "M": 1, "T": -4})

    a = torch.tensor(np.random.rand(3, 10))
    b = torch.tensor(np.random.rand(3, 10))
    c = a * b

    ap = PhlowerTensor(a, dims_1)
    bp = PhlowerTensor(b, dims_2)
    cp = ap * bp

    np.testing.assert_array_almost_equal(cp.to_tensor().numpy(), c.numpy())

    assert cp._dimension_tensor == dims_3


def test__tanh():
    a = torch.tensor(np.random.rand(3, 10))
    c = np.tanh(a)

    ap = PhlowerTensor(a)
    cp = torch.tanh(ap)

    np.testing.assert_array_almost_equal(cp.to_tensor(), c)


@pytest.mark.parametrize(
    "is_time_series, is_voxel, size, desired_rank",
    [
        (False, False, [100, 16], 0),
        (False, False, [100, 3, 16], 1),
        (False, False, [100, 3, 3, 16], 2),
        ( True, False, [4, 100, 16], 0),
        ( True, False, [4, 100, 3, 16], 1),
        ( True, False, [4, 100, 3, 3, 16], 2),
        (False,  True, [10, 10, 10, 16], 0),
        (False,  True, [10, 10, 10, 3, 16], 1),
        (False,  True, [10, 10, 10, 3, 3, 16], 2),
        ( True,  True, [4, 10, 10, 10, 16], 0),
        ( True,  True, [4, 10, 10, 10, 3, 16], 1),
        ( True,  True, [4, 10, 10, 10, 3, 3, 16], 2),
    ],
)
def test__rank(is_time_series, is_voxel, size, desired_rank):
    torch_tensor = torch.rand(*size)
    phlower_tensor = PhlowerTensor(
        torch_tensor, is_time_series=is_time_series, is_voxel=is_voxel)
    assert phlower_tensor.rank() == desired_rank


@pytest.mark.parametrize(
    "is_time_series, is_voxel, size, desired_n_vertices",
    [
        (False, False, [100, 16], 100),
        (False, False, [100, 3, 16], 100),
        (False, False, [100, 3, 3, 16], 100),
        ( True, False, [4, 100, 16], 100),
        ( True, False, [4, 100, 3, 16], 100),
        ( True, False, [4, 100, 3, 3, 16], 100),
        (False,  True, [10, 10, 10, 16], 1000),
        (False,  True, [10, 10, 10, 3, 16], 1000),
        (False,  True, [10, 10, 10, 3, 3, 16], 1000),
        ( True,  True, [4, 10, 10, 10, 16], 1000),
        ( True,  True, [4, 10, 10, 10, 3, 16], 1000),
        ( True,  True, [4, 10, 10, 10, 3, 3, 16], 1000),
    ],
)
def test__n_vertices(is_time_series, is_voxel, size, desired_n_vertices):
    torch_tensor = torch.rand(*size)
    phlower_tensor = PhlowerTensor(
        torch_tensor, is_time_series=is_time_series, is_voxel=is_voxel)
    assert phlower_tensor.n_vertices() == desired_n_vertices


def test__raises_phlower_sparse_rank_undefined_error():
    torch_sparse_tensor = torch.eye(5).to_sparse()
    phlower_sparse_tensor = PhlowerTensor(torch_sparse_tensor)
    with pytest.raises(PhlowerSparseUnsupportedError):
        phlower_sparse_tensor.rank()


@pytest.mark.parametrize(
    "is_time_series, is_voxel, size, desired_shape",
    [
        (False, False, [100, 16], (100, 16)),
        (False, False, [100, 3, 16], (100, 3 * 16)),
        (False, False, [100, 3, 3, 16], (100, 3 * 3 * 16)),
        ( True, False, [4, 100, 16], (100, 4 * 16)),
        ( True, False, [4, 100, 3, 16], (100, 4 * 3 * 16)),
        ( True, False, [4, 100, 3, 3, 16], (100, 4 * 3 * 3 * 16)),
        (False,  True, [10, 10, 10, 16], (1000, 16)),
        (False,  True, [10, 10, 10, 3, 16], (1000, 3 * 16)),
        (False,  True, [10, 10, 10, 3, 3, 16], (1000, 3 * 3 * 16)),
        ( True,  True, [4, 10, 10, 10, 16], (1000, 4 * 16)),
        ( True,  True, [4, 10, 10, 10, 3, 16], (1000, 4 * 3 * 16)),
        ( True,  True, [4, 10, 10, 10, 3, 3, 16], (1000, 4 * 3 * 3 * 16)),
    ],
)
def test__to_vertexwise(is_time_series, is_voxel, size, desired_shape):
    torch_tensor = torch.rand(*size)
    phlower_tensor = PhlowerTensor(
        torch_tensor, is_time_series=is_time_series, is_voxel=is_voxel)
    assert phlower_tensor.to_vertexwise()[0].shape == desired_shape


@pytest.mark.parametrize(
    "is_time_series, is_voxel, size",
    [
        (False, False, [100, 16]),
        (False, False, [100, 3, 16]),
        (False, False, [100, 3, 3, 16]),
        ( True, False, [4, 100, 16]),
        ( True, False, [4, 100, 3, 16]),
        ( True, False, [4, 100, 3, 3, 16]),
        (False,  True, [10, 10, 10, 16]),
        (False,  True, [10, 10, 10, 3, 16]),
        (False,  True, [10, 10, 10, 3, 3, 16]),
        ( True,  True, [4, 10, 10, 10, 16]),
        ( True,  True, [4, 10, 10, 10, 3, 16]),
        ( True,  True, [4, 10, 10, 10, 3, 3, 16]),
    ],
)
def test__to_vertexwise_inverse(is_time_series, is_voxel, size):
    torch_tensor = torch.rand(*size)
    phlower_tensor = PhlowerTensor(
        torch_tensor, is_time_series=is_time_series, is_voxel=is_voxel)
    vertexwise, original_pattern, resultant_pattern, dict_shape\
        = phlower_tensor.to_vertexwise()
    assert len(vertexwise.shape) == 2
    pattern = f"{resultant_pattern} -> {original_pattern}"
    actual = vertexwise.rearrange(
        pattern, is_time_series=is_time_series, is_voxel=is_voxel,
        **dict_shape)
    np.testing.assert_almost_equal(
        actual.to_tensor().numpy(),
        phlower_tensor.to_tensor().numpy())


@pytest.mark.parametrize(
    "input_shape, pattern, dict_shape, desired_shape",
    [
        ((10, 3, 16), "n p a -> n (p a)", {"a": 16}, (10, 3 * 16)),
        ((10, 3 * 16), "n (p a) -> n p a", {"p": 3}, (10, 3, 16)),
    ],
)
def test__rearrange(input_shape, pattern, dict_shape, desired_shape):
    phlower_tensor = PhlowerTensor(torch.rand(*input_shape))
    actual = phlower_tensor.rearrange(pattern, **dict_shape)
    assert actual.shape == desired_shape
