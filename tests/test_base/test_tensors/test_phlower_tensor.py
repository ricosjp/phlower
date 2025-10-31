from collections.abc import Callable
from unittest import mock

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as extra_np
from phlower import PhlowerTensor, phlower_dimension_tensor, phlower_tensor
from phlower.utils.enums import PhysicalDimensionSymbolType
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


@given(
    data=st.lists(
        st.floats(width=32, allow_infinity=False, allow_nan=False),
        min_size=1,
        max_size=100,
    )
)
def test__overwrite_time_series_flag(data: list[float]):
    a = phlower_tensor(
        data,
        is_time_series=False,
        is_voxel=False,
        dimension={"T": 1, "L": -2},
    )
    b = phlower_tensor(torch.stack([a, a, a]), is_time_series=True)
    assert b.is_time_series
    assert a.dimension == b.dimension


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


@st.composite
def random_phlower_tensors_with_same_dimension_and_shape(
    draw: Callable, shape: tuple[int] | st.SearchStrategy[int]
) -> tuple[PhlowerTensor, PhlowerTensor]:
    _shape = draw(shape)

    array1 = draw(
        extra_np.arrays(
            dtype=np.dtypes.Float32DType(),
            shape=_shape,
        )
    )
    array2 = draw(
        extra_np.arrays(
            dtype=np.dtypes.Float32DType(),
            shape=_shape,
        )
    )

    dimensions = draw(
        st.lists(
            elements=st.floats(allow_nan=False, allow_infinity=False),
            min_size=len(PhysicalDimensionSymbolType),
            max_size=len(PhysicalDimensionSymbolType),
        )
    )

    return (
        phlower_tensor(torch.from_numpy(array1), dimension=dimensions),
        phlower_tensor(torch.from_numpy(array2), dimension=dimensions),
    )


@given(
    random_phlower_tensors_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        )
    )
)
def test__add_tensor_content_with_dimension(
    tensors: tuple[PhlowerTensor, PhlowerTensor],
):
    a, b = tensors
    c = a.to_numpy() + b.to_numpy()

    cp = a + b
    np.testing.assert_array_almost_equal(cp.to_numpy(), c)

    assert cp.dimension == a.dimension


@given(
    random_phlower_tensors_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        )
    )
)
def test__sub_tensor_content_with_dimension(
    tensors: tuple[PhlowerTensor, PhlowerTensor],
):
    a, b = tensors

    c = a.to_numpy() - b.to_numpy()
    cp = a - b

    np.testing.assert_array_almost_equal(cp.to_numpy(), c)

    assert cp.dimension == a.dimension


@given(
    shape=st.lists(
        st.integers(min_value=1, max_value=10), min_size=1, max_size=5
    )
)
@pytest.mark.parametrize("dimension", [None, {}])
def test__rsub_tensor_content_with_dimension(
    shape: tuple[int, ...],
    dimension: dict | None,
):
    a = phlower_tensor(torch.rand(shape), dimension=dimension)
    b = torch.rand(shape)

    cp = b - a
    desired = b.numpy() - a.numpy()

    np.testing.assert_array_almost_equal(cp.numpy(), desired)

    assert cp.dimension == a.dimension


@given(
    random_phlower_tensors_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        )
    )
)
def test__neg_with_unit(tensors: tuple[PhlowerTensor, PhlowerTensor]):
    a, _ = tensors

    c = -a.to_numpy()
    cp = -a

    np.testing.assert_array_almost_equal(cp.to_numpy(), c)
    assert cp.dimension == a.dimension


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


@given(
    scalar_array=st.floats(allow_nan=False, allow_infinity=False, width=64),
    time_series_array=st.lists(
        st.floats(allow_nan=False, allow_infinity=False, width=64),
        min_size=2,
        max_size=5,
    ),
)
def test__add_scalar_and_timeseries(
    scalar_array: float, time_series_array: list[float]
):
    a = phlower_tensor([scalar_array], is_time_series=False, is_voxel=False)
    b = phlower_tensor(time_series_array, is_time_series=True, is_voxel=False)
    assert not a.is_time_series
    assert b.is_time_series
    assert (a + b).is_time_series
    assert (b + a).is_time_series


@pytest.mark.parametrize(
    "slicer, tensor_shape, is_time_series",
    [
        (slice(2, 15, 3), (20, 3, 1), True),
        (slice(0, 60, 5), (60, 3, 1), True),
        (slice(None, None, 5), (60, 3, 1), True),
        ([2, 5, 8], (20, 3, 1), True),
        (0, (20, 3, 1), False),
        (slice(2, None, 3), (20, 3, 1), True),
        (59, (60, 3, 1), False),
    ],
)
@given(
    dimensions=st.lists(
        elements=st.floats(allow_nan=False, allow_infinity=False, width=32),
        min_size=len(PhysicalDimensionSymbolType),
        max_size=len(PhysicalDimensionSymbolType),
    )
)
def test__slice_time_with_slicable_object(
    slicer: slice | list | int,
    tensor_shape: tuple,
    is_time_series: bool,
    dimensions: list[float],
):
    a = phlower_tensor(
        np.random.rand(*tensor_shape),
        is_time_series=True,
        is_voxel=False,
        dimension=dimensions,
    )
    b = a.slice_time(slicer)
    assert b.is_time_series is is_time_series
    assert b.dimension == a.dimension
    np.testing.assert_almost_equal(b.numpy(), a.to_tensor()[[slicer]].numpy())


@pytest.mark.parametrize(
    "dim_1, dim_2, desired_dim",
    [({"L": 2, "T": -2}, {"M": 1, "T": -2}, {"L": 2, "M": 1, "T": -4})],
)
@given(
    input_shape=st.lists(
        st.integers(min_value=1, max_value=10), min_size=1, max_size=5
    )
)
def test__mul_with_unit(
    dim_1: dict, dim_2: dict, desired_dim: dict, input_shape: list[int]
):
    ap = phlower_tensor(
        torch.tensor(np.random.rand(*input_shape)), dimension=dim_1
    )
    bp = phlower_tensor(
        torch.tensor(np.random.rand(*input_shape)), dimension=dim_2
    )

    c = ap.to_numpy() * bp.to_numpy()
    cp = ap * bp

    np.testing.assert_array_almost_equal(cp.to_numpy(), c)

    assert cp._dimension_tensor == phlower_dimension_tensor(desired_dim)


@pytest.mark.parametrize(
    "dim_1, dim_2, desired_dim",
    [({"L": 2, "T": 2}, {"M": 1, "T": 2}, {"L": 2, "M": -1, "T": 0})],
)
@given(
    input_shape=st.lists(
        st.integers(min_value=1, max_value=10), min_size=1, max_size=5
    )
)
def test__div_with_unit(
    dim_1: dict, dim_2: dict, desired_dim: dict, input_shape: list[int]
):
    ap = phlower_tensor(
        torch.tensor(np.random.rand(*input_shape)), dimension=dim_1
    )
    bp = phlower_tensor(
        1.0 + torch.tensor(np.random.rand(*input_shape)), dimension=dim_2
    )

    c = ap.to_numpy() / bp.to_numpy()
    cp = ap / bp

    np.testing.assert_array_almost_equal(cp.to_numpy(), c)

    assert cp._dimension_tensor == phlower_dimension_tensor(desired_dim)


@given(
    random_phlower_tensors_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        )
    )
)
def test__tensor_div_scalar(tensors: tuple[PhlowerTensor, PhlowerTensor]):
    a, _ = tensors

    c = a.to_numpy() / 3.0
    cp: PhlowerTensor = a / 3.0

    np.testing.assert_array_almost_equal(cp.to_numpy(), c)

    assert cp.dimension == a.dimension


@pytest.mark.parametrize(
    "dim_1, desired_dim",
    [({"L": 2, "T": 2}, {"L": -2, "T": -2}), ({"Theta": 5}, {"Theta": -5})],
)
@given(
    input_shape=st.lists(
        st.integers(min_value=1, max_value=10), min_size=1, max_size=5
    )
)
def test__scalar_div_tensor(
    dim_1: dict, desired_dim: dict, input_shape: list[int]
):
    ap = phlower_tensor(
        (1.0 + torch.tensor(np.random.rand(*input_shape))), dimension=dim_1
    )

    c = 3.0 / ap.to_numpy()
    cp = 3.0 / ap
    np.testing.assert_array_almost_equal(cp.to_numpy(), c)

    assert cp.dimension == phlower_dimension_tensor(desired_dim)


def test__exp():
    a = torch.tensor(np.random.rand(3, 10))
    c = np.exp(a)

    ap = phlower_tensor(a)
    cp = torch.exp(ap)

    np.testing.assert_array_almost_equal(cp.to_tensor(), c)


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
    "input_shape, is_time_series, is_voxel, pattern, dict_shape, desired_shape",
    [
        (
            (10, 3, 16),
            False,
            False,
            "n p a -> n (p a)",
            {"a": 16},
            (10, 3 * 16),
        ),
        ((10, 3 * 16), False, False, "n (p a) -> n p a", {"p": 3}, (10, 3, 16)),
        [
            (10, 8, 8, 8, 3, 3, 16),
            True,
            True,
            "... (f g) -> ... f g",
            {"f": 4},
            (10, 8, 8, 8, 3, 3, 4, 4),
        ],
        [
            (10, 8, 8, 8, 3, 3, 16),
            True,
            False,
            "t ... (f g) -> t ... f g",
            {"f": 4},
            (10, 8, 8, 8, 3, 3, 4, 4),
        ],
    ],
)
def test__rearrange(
    input_shape: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    pattern: str,
    dict_shape: dict,
    desired_shape: tuple[int],
):
    ph_tensor = phlower_tensor(
        torch.rand(*input_shape),
        is_time_series=is_time_series,
        is_voxel=is_voxel,
    )
    actual = ph_tensor.rearrange(pattern, **dict_shape)
    assert actual.shape == desired_shape
    assert actual.is_time_series is ph_tensor.is_time_series
    assert actual.is_voxel is ph_tensor.is_voxel


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


@pytest.mark.parametrize("dim", [0, -1])
@given(
    random_phlower_tensors_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        )
    )
)
def test__stack_operation(
    dim: int, tensors: tuple[PhlowerTensor, PhlowerTensor]
):
    a, b = tensors
    stacked_pht: PhlowerTensor = torch.stack([a, b], dim=dim)

    assert stacked_pht.dimension == a.dimension

    torch_tensor = torch.stack([a.to_tensor(), b.to_tensor()], dim=dim)
    assert stacked_pht.shape == torch_tensor.shape


@pytest.mark.parametrize("op", [torch.max, torch.min])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4)])
@pytest.mark.parametrize("dimension", [None, {"L": 3}])
def test__min_max_operation_with_dim(
    op: callable,
    dim: int,
    shape: tuple[int],
    dimension: dict[str, int] | None,
):
    torch_tensor = torch.rand(shape)
    tensor = phlower_tensor(torch_tensor, dimension=dimension)
    ret = op(tensor, dim=dim)
    desired = op(torch_tensor, dim=dim)
    assert ret.values.dimension == tensor.dimension
    np.testing.assert_almost_equal(ret.values.numpy(), desired.values.numpy())
    np.testing.assert_almost_equal(ret.indices.numpy(), desired.indices.numpy())
    if dimension is None:
        assert ret.indices.dimension is None
    else:
        assert ret.indices.dimension.is_dimensionless


@pytest.mark.parametrize("dimension", [None, {}, {"L": 3}])
def test__setitem_with_scalar(dimension: dict[str, int] | None):
    t = phlower_tensor(torch.rand(5), dimension=dimension)
    s = np.random.rand()
    t[2:] = s

    np.testing.assert_almost_equal(t[2:].numpy(), s)

    if dimension is None:
        assert t.dimension is None
    else:
        assert t.dimension == phlower_dimension_tensor(dimension)


@pytest.mark.parametrize("dimension", [None, {}, {"L": 3}])
def test__setitem_with_phlower_tensor(dimension: dict[str, int] | None):
    t = phlower_tensor(torch.rand(5), dimension=dimension)
    s = phlower_tensor(torch.rand(5), dimension=dimension)
    t[2:] = s[2:]

    np.testing.assert_almost_equal(t[2:].numpy(), s[2:].numpy())

    if dimension is None:
        assert t.dimension is None
    else:
        assert t.dimension == phlower_dimension_tensor(dimension)


@pytest.mark.parametrize("dimension", [None, {}, {"L": 3}])
def test__setitem_raise_dimension_incompatible_error(
    dimension: dict[str, int] | None,
):
    t = phlower_tensor(torch.rand(5), dimension=dimension)
    s = phlower_tensor(torch.rand(5), dimension={"L": 2})
    with pytest.raises(DimensionIncompatibleError):
        t[2:] = s[2:]


# region Test for __getitem__


@pytest.mark.parametrize(
    "shape, index, desired_time_series",
    [
        ((5, 3, 4), 0, False),
        (
            (5, 3, 4),
            torch.tensor([1, 2, 3], dtype=torch.long),
            True,
        ),
        (
            (4, 4, 4, 3, 4),
            (torch.rand(4, 4, 4, 3, 4) > 0.5),
            False,
        ),
        (
            (4, 4, 4, 3, 4),
            np.array([1, 2, 3], dtype=np.long),
            True,
        ),
        (
            (5, 3, 4),
            [...],
            True,
        ),
        ((5, 3, 4), [slice(1, 5)], True),
        ((5, 3, 4), [slice(1, 2)], True),
        ((5, 3, 4), [None, ...], False),
        ((5, 3, 4), [..., None], True),
        ((3, 6, 3, 3, 1), [..., None, slice(1, 2, 1)], True),
    ],
)
def test__getitem_for_timeseries_phlower_tensor(
    shape: tuple[int],
    index: int,
    desired_time_series: bool,
):
    base_tensor = torch.rand(*shape)
    pht = phlower_tensor(base_tensor, is_time_series=True)
    actual = pht[index]
    assert actual.is_time_series == desired_time_series

    np.testing.assert_array_almost_equal(
        actual.to_numpy(), base_tensor[index].numpy()
    )


@pytest.mark.parametrize(
    "shape, index, desired_voxel",
    [
        ((4, 4, 4, 3, 4), 0, False),
        ((4, 4, 4, 3, 4), torch.tensor([1, 2, 3], dtype=torch.long), True),
        (
            (4, 4, 4, 3, 4),
            (torch.rand(4, 4, 4, 3, 4) > 0.5),
            False,
        ),
        (
            (4, 4, 4, 3, 4),
            np.array([1, 2, 3], dtype=np.long),
            True,
        ),
        ((4, 4, 4, 3, 4), [..., 0], True),
        ((5, 4, 5, 3, 4), [None, ..., 0], False),
        ((5, 4, 5, 3, 4), [slice(1, 2)], True),
        ((5, 4, 5, 3, 4), [slice(1, 5), slice(1, 5), slice(1, 5)], True),
        ((5, 4, 5, 3, 4), [slice(1, 2), slice(1, 5), slice(1, 5)], True),
        ((5, 4, 5, 3, 4), [slice(1, 5), slice(1, 2), slice(1, 5)], True),
        ((5, 4, 5, 3, 4), [slice(1, 5), slice(1, 5), slice(1, 2)], True),
        ((5, 4, 5, 3, 4), [slice(1, 5), ..., None, slice(1, 5)], True),
    ],
)
def test__getitem_for_voxel_phlower_tensor(
    shape: tuple[int],
    index: int,
    desired_voxel: bool,
):
    base_tensor = torch.rand(*shape)
    pht = phlower_tensor(base_tensor, is_voxel=True)
    actual = pht[index]
    assert actual.is_voxel == desired_voxel

    np.testing.assert_array_almost_equal(
        actual.to_numpy(), base_tensor[index].numpy()
    )


@pytest.mark.parametrize(
    "shape, index, desired_time_series, desired_voxel",
    [
        ((4, 4, 4, 3, 4), 0, False, True),
        (
            (4, 4, 4, 3, 4),
            torch.tensor([1, 2, 3], dtype=torch.long),
            True,
            True,
        ),
        (
            (4, 4, 4, 3, 4),
            (torch.rand(4, 4, 4, 3, 4) > 0.5),
            False,
            False,
        ),
        (
            (4, 4, 4, 3, 4),
            np.array([1, 2, 3], dtype=np.long),
            True,
            True,
        ),
        ((4, 4, 4, 3, 4), [..., 0], True, True),
        ((5, 4, 5, 3, 4), [None, ..., 0], False, False),
        ((5, 4, 5, 3, 4), [..., None, 0], True, True),
        ((5, 4, 5, 3, 4), slice(1, 2), True, True),
        ((5, 4, 5, 3, 4), slice(1, 3, 3), True, True),
        ((5, 4, 5, 3, 4), [slice(1, 5), slice(1, 5), slice(1, 5)], True, True),
        ((5, 4, 5, 3, 4), [slice(1, 2), slice(1, 5), slice(1, 5)], True, True),
        ((5, 4, 5, 3, 4), [slice(1, 5), slice(1, 2), slice(1, 5)], True, True),
        ((5, 4, 5, 3, 4), [slice(1, 5), slice(1, 5), slice(1, 2)], True, True),
        (
            (5, 4, 5, 3, 4),
            [slice(1, 5), slice(1, 5), slice(1, 4), slice(1, 2)],
            True,
            True,
        ),
        ((5, 4, 5, 3, 4), [slice(1, 5), ..., None, slice(1, 5)], True, True),
    ],
)
def test__getitem_for_voxel_and_timeseries_phlower_tensor(
    shape: tuple[int],
    index: int,
    desired_time_series: bool,
    desired_voxel: bool,
):
    base_tensor = torch.rand(*shape)
    pht = phlower_tensor(base_tensor, is_voxel=True, is_time_series=True)
    actual = pht[index]
    assert actual.is_time_series == desired_time_series
    assert actual.is_voxel == desired_voxel

    np.testing.assert_array_almost_equal(
        actual.to_numpy(), base_tensor[index].numpy()
    )


@pytest.mark.parametrize(
    "shape, index",
    [
        ((4, 4, 4, 3, 4), 0),
        (
            (4, 4, 4, 3, 4),
            torch.tensor([1, 2, 3], dtype=torch.long),
        ),
        ((4, 4, 4, 3, 4), [..., 0]),
        ((5, 4, 5, 3, 4), [None, ..., 0]),
        ((5, 4, 5, 3, 4), slice(1, 3, 3)),
        ((5, 4, 5, 3, 4), [slice(1, 5), ..., None, slice(1, 5)]),
    ],
)
def test__getitem_for_nodal_phlower_tensor(
    shape: tuple[int],
    index: int,
):
    base_tensor = torch.rand(*shape)
    pht = phlower_tensor(base_tensor)
    actual = pht[index]
    assert actual.is_time_series is False
    assert actual.is_voxel is False

    np.testing.assert_array_almost_equal(
        actual.to_numpy(), base_tensor[index].numpy()
    )


# endregion
