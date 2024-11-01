from unittest import mock

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from phlower import PhlowerTensor, phlower_dimension_tensor, phlower_tensor
from phlower.utils.exceptions import (
    DimensionIncompatibleError,
    PhlowerSparseUnsupportedError,
)


@given(st.lists(st.floats(width=32), min_size=1, max_size=100))
def test__create_default_phlower_tensor(values: list[float]):
    pht = phlower_tensor(values)
    assert pht.is_time_series is False
    assert pht.is_voxel is False


def test__create_same_initialized_object_from_list_and_tensor():
    list_data = [0.1, 0.2, 0.3]
    pht_list = phlower_tensor(list_data)
    pht_torch = phlower_tensor(torch.tensor(list_data))
    np.testing.assert_array_almost_equal(
        pht_list.to_numpy(), pht_torch.to_numpy()
    )


@pytest.mark.parametrize(
    "device",
    [torch.device("cpu"), torch.device("meta"), "cpu", "meta"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64],
)
def test__check_dtype_and_device_after_applying_to(
    device: torch.device | str, dtype: torch.dtype
):
    pht: PhlowerTensor = phlower_tensor(
        [0.1, 0.2, 0.3], dimension={"L": 2, "T": -1}
    )

    converted_pht = pht.to(device=device, dtype=dtype)
    assert converted_pht.device.type == str(device)
    assert converted_pht.dimension.device.type == str(device)
    assert converted_pht.dtype == dtype
    assert converted_pht.dimension.dtype == dtype


@pytest.mark.parametrize(
    "device",
    ["cpu", "meta", "cuda:0", "cuda:1"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.float32, torch.float64],
)
def test__pass_arguments_to_torch_function(device: str, dtype: torch.dtype):
    pht: PhlowerTensor = phlower_tensor([0.1, 0.2, 0.3], dimension=None)

    with mock.patch.object(torch.Tensor, "to") as mocked:
        mocked.return_value = pht._tensor

        _ = pht.to(device=device, dtype=dtype)

        assert mocked.call_count == 1

        for args in mocked.call_args_list:
            assert args.kwargs.get("device") == device
            assert args.kwargs.get("dtype") == dtype


def test__to_numpy_same_as_numpy():
    pht = phlower_tensor([0.1, 0.2, 0.3], dimension={"L": 2, "T": -1})
    np.testing.assert_array_almost_equal(pht.numpy(), pht.to_numpy())


def test__from_pattern():
    pht = phlower_tensor([0.1, 0.2, 0.3], dimension={"L": 2, "T": -1})
    pht_from_pattern = PhlowerTensor.from_pattern(
        pht.to_tensor(), pht.dimension, pht.shape_pattern
    )

    np.testing.assert_array_almost_equal(pht_from_pattern.numpy(), pht.numpy())
    np.testing.assert_array_almost_equal(
        pht_from_pattern.dimension._tensor.numpy(),
        pht.dimension._tensor.numpy(),
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


def test__sub_with_unit():
    units = phlower_dimension_tensor({"L": 2, "T": -2})
    a = np.random.rand(3, 10)
    b = np.random.rand(3, 10)
    c = a - b

    ap = PhlowerTensor(torch.tensor(a), units)
    bp = PhlowerTensor(torch.tensor(b), units)
    cp = ap - bp

    np.testing.assert_array_almost_equal(cp.to_tensor().numpy(), c)


def test__neg_with_unit():
    units = phlower_dimension_tensor({"L": 2, "T": -2})
    a = np.random.rand(3, 10)
    c = -a

    ap = PhlowerTensor(torch.tensor(a), units)
    cp = -ap

    np.testing.assert_array_almost_equal(cp.numpy(), c)


@pytest.mark.parametrize(
    "unit1, unit2", [({"L": 2, "T": -2}, None), (None, {"M": 2, "T": -3})]
)
def test__add_with_and_without_dimensions(
    unit1: dict | None, unit2: dict | None
):
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


def test__div_with_unit():
    dims_1 = phlower_dimension_tensor({"L": 2, "T": -2})
    dims_2 = phlower_dimension_tensor({"M": 1, "T": -2})
    dims_3 = phlower_dimension_tensor({"L": 2, "M": -1, "T": 0})

    a = torch.tensor(np.random.rand(3, 10))
    b = torch.tensor(np.random.rand(3, 10))
    c = a / b

    ap = PhlowerTensor(a, dims_1)
    bp = PhlowerTensor(b, dims_2)
    cp = ap / bp

    np.testing.assert_array_almost_equal(cp.to_tensor().numpy(), c.numpy())

    assert cp._dimension_tensor == dims_3


def test__tensor_div_scalar():
    dims = phlower_dimension_tensor({"L": 2, "T": -2})

    a = torch.tensor(np.random.rand(3, 10))
    c = a / 3.0

    ap = PhlowerTensor(a, dims)
    cp: PhlowerTensor = ap / 3.0

    np.testing.assert_array_almost_equal(cp.to_tensor().numpy(), c.numpy())

    assert cp.dimension == dims


def test__scalar_div_tensor():
    dims = phlower_dimension_tensor({"L": 2, "T": -2})
    desired_dims = phlower_dimension_tensor({"L": -2, "T": 2})

    a = torch.tensor(np.random.rand(3, 10))
    c = 3.0 / a

    ap = PhlowerTensor(a, dims)
    cp: PhlowerTensor = 3.0 / ap

    np.testing.assert_array_almost_equal(cp.to_tensor().numpy(), c.numpy())

    assert cp.dimension == desired_dims


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
        (True, False, [4, 100, 16], 0),
        (True, False, [4, 100, 3, 16], 1),
        (True, False, [4, 100, 3, 3, 16], 2),
        (False, True, [10, 10, 10, 16], 0),
        (False, True, [10, 10, 10, 3, 16], 1),
        (False, True, [10, 10, 10, 3, 3, 16], 2),
        (True, True, [4, 10, 10, 10, 16], 0),
        (True, True, [4, 10, 10, 10, 3, 16], 1),
        (True, True, [4, 10, 10, 10, 3, 3, 16], 2),
    ],
)
def test__rank(
    is_time_series: bool, is_voxel: bool, size: list[int], desired_rank: int
):
    torch_tensor = torch.rand(*size)
    phlower_tensor = PhlowerTensor(
        torch_tensor, is_time_series=is_time_series, is_voxel=is_voxel
    )
    assert phlower_tensor.rank() == desired_rank


@pytest.mark.parametrize(
    "is_time_series, is_voxel, size, desired_n_vertices",
    [
        (False, False, [100, 16], 100),
        (False, False, [100, 3, 16], 100),
        (False, False, [100, 3, 3, 16], 100),
        (True, False, [4, 100, 16], 100),
        (True, False, [4, 100, 3, 16], 100),
        (True, False, [4, 100, 3, 3, 16], 100),
        (False, True, [10, 10, 10, 16], 1000),
        (False, True, [10, 10, 10, 3, 16], 1000),
        (False, True, [10, 10, 10, 3, 3, 16], 1000),
        (True, True, [4, 10, 10, 10, 16], 1000),
        (True, True, [4, 10, 10, 10, 3, 16], 1000),
        (True, True, [4, 10, 10, 10, 3, 3, 16], 1000),
    ],
)
def test__n_vertices(
    is_time_series: bool,
    is_voxel: bool,
    size: list[int],
    desired_n_vertices: int,
):
    torch_tensor = torch.rand(*size)
    phlower_tensor = PhlowerTensor(
        torch_tensor, is_time_series=is_time_series, is_voxel=is_voxel
    )
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
        (True, False, [4, 100, 16], (100, 4 * 16)),
        (True, False, [4, 100, 3, 16], (100, 4 * 3 * 16)),
        (True, False, [4, 100, 3, 3, 16], (100, 4 * 3 * 3 * 16)),
        (False, True, [10, 10, 10, 16], (1000, 16)),
        (False, True, [10, 10, 10, 3, 16], (1000, 3 * 16)),
        (False, True, [10, 10, 10, 3, 3, 16], (1000, 3 * 3 * 16)),
        (True, True, [4, 10, 10, 10, 16], (1000, 4 * 16)),
        (True, True, [4, 10, 10, 10, 3, 16], (1000, 4 * 3 * 16)),
        (True, True, [4, 10, 10, 10, 3, 3, 16], (1000, 4 * 3 * 3 * 16)),
    ],
)
def test__to_vertexwise(
    is_time_series: bool,
    is_voxel: bool,
    size: list[int],
    desired_shape: tuple[int],
):
    torch_tensor = torch.rand(*size)
    phlower_tensor = PhlowerTensor(
        torch_tensor, is_time_series=is_time_series, is_voxel=is_voxel
    )
    assert phlower_tensor.to_vertexwise()[0].shape == desired_shape


@pytest.mark.parametrize(
    "is_time_series, is_voxel, size",
    [
        (False, False, [100, 16]),
        (False, False, [100, 3, 16]),
        (False, False, [100, 3, 3, 16]),
        (True, False, [4, 100, 16]),
        (True, False, [4, 100, 3, 16]),
        (True, False, [4, 100, 3, 3, 16]),
        (False, True, [10, 10, 10, 16]),
        (False, True, [10, 10, 10, 3, 16]),
        (False, True, [10, 10, 10, 3, 3, 16]),
        (True, True, [4, 10, 10, 10, 16]),
        (True, True, [4, 10, 10, 10, 3, 16]),
        (True, True, [4, 10, 10, 10, 3, 3, 16]),
    ],
)
def test__to_vertexwise_inverse(
    is_time_series: bool, is_voxel: bool, size: list[int]
):
    torch_tensor = torch.rand(*size)
    phlower_tensor = PhlowerTensor(
        torch_tensor, is_time_series=is_time_series, is_voxel=is_voxel
    )
    vertexwise, resultant_pattern = phlower_tensor.to_vertexwise()
    assert len(vertexwise.shape) == 2
    pattern = (
        f"{resultant_pattern} -> {phlower_tensor.shape_pattern.get_pattern()}"
    )
    dict_shape = phlower_tensor.shape_pattern.get_pattern_to_size(
        drop_last=True
    )
    actual = vertexwise.rearrange(pattern, **dict_shape)
    np.testing.assert_almost_equal(
        actual.to_tensor().numpy(), phlower_tensor.to_tensor().numpy()
    )


@pytest.mark.parametrize(
    "input_shape, pattern, dict_shape, desired_shape",
    [
        ((10, 3, 16), "n p a -> n (p a)", {"a": 16}, (10, 3 * 16)),
        ((10, 3 * 16), "n (p a) -> n p a", {"p": 3}, (10, 3, 16)),
    ],
)
def test__rearrange(
    input_shape: tuple[int],
    pattern: str,
    dict_shape: dict,
    desired_shape: tuple[int],
):
    phlower_tensor = PhlowerTensor(torch.rand(*input_shape))
    actual = phlower_tensor.rearrange(pattern, **dict_shape)
    assert actual.shape == desired_shape


def test__clone():
    original_dimension_dict = {"L": 2, "T": -1}
    pht = phlower_tensor([0.1, 0.2, 0.3], dimension=original_dimension_dict)
    cloned = pht.clone()
    pht._tensor[1] = 10.0
    pht._dimension_tensor = pht.dimension * pht.dimension
    np.testing.assert_array_almost_equal(
        pht.numpy()[[0, 2]], cloned.numpy()[[0, 2]]
    )
    assert pht.numpy()[1] != cloned.numpy()[1]

    for k, v in original_dimension_dict.items():
        assert cloned.dimension.to_dict()[k] == v
        assert pht.dimension.to_dict()[k] == 2 * v


@pytest.mark.parametrize(
    "inputs, nan",
    [
        ([0.1, 0.2, float("nan")], 0.0),
        ([float("nan"), 0.2, float("nan")], 10.0),
    ],
)
def test__nan_to_num(inputs: list[float], nan: float):
    tensor = phlower_tensor(inputs)
    new_tensor: PhlowerTensor = torch.nan_to_num(tensor, nan=nan)
    tensor[torch.isnan(tensor)] = nan

    assert not torch.any(torch.isnan(new_tensor.to_tensor()))

    np.testing.assert_array_almost_equal(new_tensor.numpy(), tensor.numpy())
