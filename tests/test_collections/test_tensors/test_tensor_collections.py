from collections.abc import Callable

import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis.extra import numpy as extra_np
from phlower._base import PhlowerTensor, phlower_tensor
from phlower.collections import phlower_tensor_collection
from phlower.utils.enums import PhysicalDimensionSymbolType


@st.composite
def phlower_tensors_with_same_dimension(
    draw: Callable,
    shape: tuple[int] | st.SearchStrategy[int],
    has_dimension: bool = True,
    n_items: int = 2,
) -> list[PhlowerTensor]:
    arrays = [
        draw(
            extra_np.arrays(
                dtype=np.dtypes.Float32DType(),
                shape=shape,
            )
        )
        for _ in range(n_items)
    ]

    if has_dimension:
        dimensions = draw(
            st.lists(
                elements=st.floats(allow_nan=False, allow_infinity=False),
                min_size=len(PhysicalDimensionSymbolType),
                max_size=len(PhysicalDimensionSymbolType),
            )
        )
    else:
        dimensions = [0 for _ in range(len(PhysicalDimensionSymbolType))]

    return [phlower_tensor(arr, dimension=dimensions) for arr in arrays]


# region Test for comparison


@given(
    st.lists(
        extra_np.arrays(
            np.float32,
            1,
            elements=st.floats(
                min_value=-1000,
                max_value=1000,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        ),
        min_size=1,
        max_size=20,
    )
)
def test__comparison_with_the_other_collection(values: list[np.ndarray]):
    dict_data = phlower_tensor_collection(
        {f"key_{i}": v for i, v in enumerate(values)}
    )
    less_dict_data = dict_data - 1.0
    large_dict_data = dict_data + 1.0

    assert dict_data > less_dict_data
    assert dict_data >= less_dict_data

    assert dict_data >= dict_data
    assert dict_data <= dict_data

    assert large_dict_data > dict_data
    assert large_dict_data >= dict_data


@given(
    st.lists(
        extra_np.arrays(
            np.float32,
            1,
            elements=st.floats(
                min_value=-1000,
                max_value=1000,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        ),
        min_size=1,
        max_size=20,
    )
)
def test__comparison_with_less_float_value(values: list[np.ndarray]):
    dict_data = phlower_tensor_collection(
        {f"key_{i}": v for i, v in enumerate(values)}
    )

    assert dict_data >= -1000.0
    assert dict_data > -1001.0


@given(
    st.lists(
        extra_np.arrays(
            np.float32,
            1,
            elements=st.floats(
                min_value=-1000,
                max_value=1000,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
        ),
        min_size=1,
        max_size=20,
    )
)
def test__comparison_with_greater_float_value(values: list[np.ndarray]):
    dict_data = phlower_tensor_collection(
        {f"key_{i}": v for i, v in enumerate(values)}
    )

    assert dict_data <= 1000.0
    assert dict_data < 1001.0


@given(
    phlower_tensors_with_same_dimension(
        shape=(3, 4), has_dimension=False, n_items=5
    )
)
def test__raise_error_when_cannot_compare(values: list[PhlowerTensor]):
    dict_data = phlower_tensor_collection(
        {f"key_{i}": v for i, v in enumerate(values)}
    )

    less_dict_data = dict_data - 1.0

    with pytest.raises(RuntimeError) as ex:
        _ = dict_data > less_dict_data

    assert (
        "Boolean value of Tensor with more than one value is ambiguous"
        in str(ex)
    )


#  endregion


# region Test for add


@given(
    phlower_tensors_with_same_dimension(shape=(3, 4)),
    phlower_tensors_with_same_dimension(shape=(5, 11, 2)),
)
@settings(deadline=600)
def test__add_tensor_collections(
    tensors1: tuple[PhlowerTensor, PhlowerTensor],
    tensors2: tuple[PhlowerTensor, PhlowerTensor],
):
    args = [tensors1, tensors2]
    key_names = ["sample1", "sample2"]
    dict_data1 = phlower_tensor_collection(
        {k: args[i][0] for i, k in enumerate(key_names)}
    )
    dict_data2 = phlower_tensor_collection(
        {k: args[i][1] for i, k in enumerate(key_names)}
    )

    new_dict = dict_data1 + dict_data2

    for i, k in enumerate(key_names):
        print(f"Adding {k}: {args[i][0].shape} + {args[i][1].shape}")
        np.testing.assert_array_almost_equal(
            new_dict[k], (args[i][0] + args[i][1])
        )


@pytest.mark.parametrize("bias", [(0.5), (torch.Tensor([3.0])), (1.234)])
@given(
    phlower_tensors_with_same_dimension(shape=(3, 4), has_dimension=False),
)
def test__add_tensor_collections_with_float(
    bias: float | torch.Tensor,
    tensors: tuple[PhlowerTensor, PhlowerTensor],
):
    key_names = ["sample1", "sample2"]
    dict_data1 = phlower_tensor_collection(
        {k: tensors[i] for i, k in enumerate(key_names)}
    )

    new_dict = dict_data1 + bias

    for i, k in enumerate(key_names):
        np.testing.assert_array_almost_equal(new_dict[k], (tensors[i] + bias))


@pytest.mark.parametrize(
    "key_names1, key_names2",
    [
        (["sample1", "sample2"], ["sample3", "sample4"]),
        (
            ["sample1", "sample2"],
            ["sample1", "sample4"],
        ),
        (["sample1", "sample2"], ["sample1"]),
    ],
)
@given(
    phlower_tensors_with_same_dimension(shape=(3, 4)),
    phlower_tensors_with_same_dimension(shape=(5, 11, 2)),
)
def test__cannot_add_tensor_collections_with_different_keys(
    key_names1: list[str],
    key_names2: list[str],
    tensors1: tuple[PhlowerTensor, PhlowerTensor],
    tensors2: tuple[PhlowerTensor, PhlowerTensor],
):
    args = [tensors1, tensors2]
    dict_data1 = phlower_tensor_collection(
        {k: args[i][0] for i, k in enumerate(key_names1)}
    )
    dict_data2 = phlower_tensor_collection(
        {k: args[i][1] for i, k in enumerate(key_names2)}
    )

    with pytest.raises(AssertionError) as ex:
        _ = dict_data1 + dict_data2

        assert "Not allowed to add other which has different keys" in str(ex)


# endregion

# region Test for sub


@given(
    phlower_tensors_with_same_dimension(shape=(3, 4)),
    phlower_tensors_with_same_dimension(shape=(5, 11, 2)),
)
@settings(deadline=600)
def test__sub_tensor_collections(
    tensors1: tuple[PhlowerTensor, PhlowerTensor],
    tensors2: tuple[PhlowerTensor, PhlowerTensor],
):
    args = [tensors1, tensors2]
    key_names = ["sample1", "sample2"]
    dict_data1 = phlower_tensor_collection(
        {k: args[i][0] for i, k in enumerate(key_names)}
    )
    dict_data2 = phlower_tensor_collection(
        {k: args[i][1] for i, k in enumerate(key_names)}
    )

    new_dict = dict_data1 - dict_data2

    for i, k in enumerate(key_names):
        np.testing.assert_array_almost_equal(
            new_dict[k], (args[i][0] - args[i][1])
        )


@pytest.mark.parametrize("bias", [(0.5), (torch.Tensor([3.0])), (1.234)])
@given(
    phlower_tensors_with_same_dimension(shape=(3, 4), has_dimension=False),
)
def test__sub_tensor_collections_with_float(
    bias: float | torch.Tensor,
    tensors: tuple[PhlowerTensor, PhlowerTensor],
):
    key_names = ["sample1", "sample2"]
    dict_data1 = phlower_tensor_collection(
        {k: tensors[i] for i, k in enumerate(key_names)}
    )

    new_dict = dict_data1 - bias

    for i, k in enumerate(key_names):
        np.testing.assert_array_almost_equal(new_dict[k], (tensors[i] - bias))


@pytest.mark.parametrize(
    "key_names1, key_names2",
    [
        (["sample1", "sample2"], ["sample3", "sample4"]),
        (
            ["sample1", "sample2"],
            ["sample1", "sample4"],
        ),
        (["sample1", "sample2"], ["sample1"]),
    ],
)
@given(
    phlower_tensors_with_same_dimension(shape=(3, 4)),
    phlower_tensors_with_same_dimension(shape=(5, 11, 2)),
)
def test__cannot_sub_tensor_collections_with_different_keys(
    key_names1: list[str],
    key_names2: list[str],
    tensors1: tuple[PhlowerTensor, PhlowerTensor],
    tensors2: tuple[PhlowerTensor, PhlowerTensor],
):
    args = [tensors1, tensors2]
    dict_data1 = phlower_tensor_collection(
        {k: args[i][0] for i, k in enumerate(key_names1)}
    )
    dict_data2 = phlower_tensor_collection(
        {k: args[i][1] for i, k in enumerate(key_names2)}
    )

    with pytest.raises(AssertionError) as ex:
        _ = dict_data1 - dict_data2

        assert "Not allowed to substract other which has different keys" in str(
            ex
        )


# endregion

# region Test for mul


@given(
    phlower_tensors_with_same_dimension(shape=(3, 4)),
    phlower_tensors_with_same_dimension(shape=(5, 11, 2)),
)
@settings(deadline=600)
def test__mul_tensor_collections(
    tensors1: tuple[PhlowerTensor, PhlowerTensor],
    tensors2: tuple[PhlowerTensor, PhlowerTensor],
):
    args = [tensors1, tensors2]
    key_names = ["sample1", "sample2"]
    dict_data1 = phlower_tensor_collection(
        {k: args[i][0] for i, k in enumerate(key_names)}
    )
    dict_data2 = phlower_tensor_collection(
        {k: args[i][1] for i, k in enumerate(key_names)}
    )

    new_dict = dict_data1 * dict_data2

    for i, k in enumerate(key_names):
        np.testing.assert_array_almost_equal(
            new_dict[k], (args[i][0] * args[i][1])
        )


@pytest.mark.parametrize("bias", [(0.5), (torch.Tensor([3.0])), (1.234)])
@given(
    phlower_tensors_with_same_dimension(shape=(3, 4), has_dimension=False),
)
def test__mul_tensor_collections_with_float(
    bias: float | torch.Tensor,
    tensors: tuple[PhlowerTensor, PhlowerTensor],
):
    key_names = ["sample1", "sample2"]
    dict_data1 = phlower_tensor_collection(
        {k: tensors[i] for i, k in enumerate(key_names)}
    )

    new_dict = dict_data1 * bias

    for i, k in enumerate(key_names):
        np.testing.assert_array_almost_equal(new_dict[k], (tensors[i] * bias))


@pytest.mark.parametrize(
    "key_names1, key_names2",
    [
        (["sample1", "sample2"], ["sample3", "sample4"]),
        (
            ["sample1", "sample2"],
            ["sample1", "sample4"],
        ),
        (["sample1", "sample2"], ["sample1"]),
    ],
)
@given(
    phlower_tensors_with_same_dimension(shape=(3, 4)),
    phlower_tensors_with_same_dimension(shape=(5, 11, 2)),
)
def test__cannot_mul_tensor_collections_with_different_keys(
    key_names1: list[str],
    key_names2: list[str],
    tensors1: tuple[PhlowerTensor, PhlowerTensor],
    tensors2: tuple[PhlowerTensor, PhlowerTensor],
):
    args = [tensors1, tensors2]
    dict_data1 = phlower_tensor_collection(
        {k: args[i][0] for i, k in enumerate(key_names1)}
    )
    dict_data2 = phlower_tensor_collection(
        {k: args[i][1] for i, k in enumerate(key_names2)}
    )

    with pytest.raises(AssertionError) as ex:
        _ = dict_data1 * dict_data2

        assert "Not allowed to multiple other which has different keys" in str(
            ex
        )


# endregion

# region Test for truediv


@given(
    phlower_tensors_with_same_dimension(shape=(3, 4), has_dimension=False),
    phlower_tensors_with_same_dimension(shape=(5, 11, 2), has_dimension=False),
)
@settings(deadline=600)
def test__truediv_tensor_collections(
    tensors1: tuple[PhlowerTensor, PhlowerTensor],
    tensors2: tuple[PhlowerTensor, PhlowerTensor],
):
    args = [tensors1, tensors2]
    key_names = ["sample1", "sample2"]
    dict_data1 = phlower_tensor_collection(
        {k: args[i][0] for i, k in enumerate(key_names)}
    )
    dict_data2 = phlower_tensor_collection(
        {k: args[i][1] for i, k in enumerate(key_names)}
    )

    new_dict = dict_data1 / (dict_data2 + 1e-5)

    for i, k in enumerate(key_names):
        np.testing.assert_array_almost_equal(
            new_dict[k], (args[i][0] / (args[i][1] + 1e-5))
        )


@pytest.mark.parametrize("bias", [(0.5), (torch.Tensor([3.0])), (1.234)])
@given(
    phlower_tensors_with_same_dimension(shape=(3, 4), has_dimension=False),
)
def test__truediv_tensor_collections_with_float(
    bias: float | torch.Tensor,
    tensors: tuple[PhlowerTensor, PhlowerTensor],
):
    key_names = ["sample1", "sample2"]
    dict_data1 = phlower_tensor_collection(
        {k: tensors[i] for i, k in enumerate(key_names)}
    )

    new_dict = dict_data1 / bias

    for i, k in enumerate(key_names):
        np.testing.assert_array_almost_equal(new_dict[k], (tensors[i] / bias))


@pytest.mark.parametrize(
    "key_names1, key_names2",
    [
        (["sample1", "sample2"], ["sample3", "sample4"]),
        (
            ["sample1", "sample2"],
            ["sample1", "sample4"],
        ),
        (["sample1", "sample2"], ["sample1"]),
    ],
)
@given(
    phlower_tensors_with_same_dimension(shape=(3, 4)),
    phlower_tensors_with_same_dimension(shape=(5, 11, 2)),
)
def test__cannot_truediv_tensor_collections_with_different_keys(
    key_names1: list[str],
    key_names2: list[str],
    tensors1: tuple[PhlowerTensor, PhlowerTensor],
    tensors2: tuple[PhlowerTensor, PhlowerTensor],
):
    args = [tensors1, tensors2]
    dict_data1 = phlower_tensor_collection(
        {k: args[i][0] for i, k in enumerate(key_names1)}
    )
    dict_data2 = phlower_tensor_collection(
        {k: args[i][1] for i, k in enumerate(key_names2)}
    )

    with pytest.raises(AssertionError) as ex:
        _ = dict_data1 / dict_data2

        assert "Not allowed to divide by other which has different keys" in str(
            ex
        )


# endregion


@given(
    phlower_tensors_with_same_dimension(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        )
    ),
)
@settings(deadline=600)
def test__mask_tensor_collections(tensors: list[PhlowerTensor]):
    n_items = len(tensors)
    dict_data1 = phlower_tensor_collection(
        {f"sample_{i}": v for i, v in enumerate(tensors)}
    )

    keys = list(dict_data1.keys())
    mask_index = set(np.random.randint(n_items, size=5))
    mask_keys = [keys[i] for i in mask_index]
    actual = dict_data1.mask(mask_keys)

    for k in mask_keys:
        assert k in actual

    for k in mask_keys:
        np.testing.assert_array_almost_equal(actual[k], dict_data1[k])


@pytest.mark.parametrize("operation", [torch.sum, torch.mean, torch.abs])
@given(
    phlower_tensors_with_same_dimension(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        )
    ),
)
def test__apply_operation_tensor_collections(
    operation: Callable, tensors: list[PhlowerTensor]
):
    n_items = len(tensors)
    dict_data1 = phlower_tensor_collection(
        {f"sample_{i}": v for i, v in enumerate(tensors)}
    )

    actual = dict_data1.apply(operation)

    assert len(actual) == n_items
    for k in actual.keys():
        np.testing.assert_array_almost_equal(
            actual[k].numpy(), operation(dict_data1[k]).numpy()
        )


@pytest.mark.parametrize(
    "shapes, time_series, expected",
    [
        ([(3, 10, 4), (10, 4)], [True, False], 3),
        ([(3, 5, 1), (5, 1)], [True, False], 3),
        ([(5, 1), (5, 1), (10, 1)], [False, False, False], 0),
        ([(3, 1), (5, 1), (10, 1)], [False, False, False], 0),
        ([(5, 10, 4), (5, 10, 4, 1)], [True, True], 5),
    ],
)
def test___get_time_series_length_for_time_series_tensor(
    shapes: list[tuple[int, ...]],
    time_series: list[bool],
    expected: int,
):
    tensors = [
        phlower_tensor(np.random.rand(*shape), is_time_series=ts)
        for shape, ts in zip(shapes, time_series, strict=True)
    ]
    collection = phlower_tensor_collection(
        {f"sample_{i}": t for i, t in enumerate(tensors)}
    )

    assert collection.get_time_series_length() == expected


@pytest.mark.parametrize(
    "shapes, time_series",
    [
        ([(3, 10, 4), (10, 4)], [True, True]),
        ([(5, 10, 4), (6, 10, 4), (10, 1)], [True, True, False]),
    ],
)
def test___get_time_series_length_for_invalid_time_series_tensor(
    shapes: list[tuple[int, ...]], time_series: list[bool]
):
    tensors = [
        phlower_tensor(np.random.rand(*shape), is_time_series=ts)
        for shape, ts in zip(shapes, time_series, strict=True)
    ]
    collection = phlower_tensor_collection(
        {f"sample_{i}": t for i, t in enumerate(tensors)}
    )

    with pytest.raises(ValueError) as ex:
        _ = collection.get_time_series_length()
    assert "Cannot determine time series length" in str(ex)


@pytest.mark.parametrize(
    "shapes, time_series",
    [
        ([(3, 10, 4), (10, 4)], [True, False]),
        ([(3, 5, 1), (5, 1)], [True, False]),
        ([(5, 1), (5, 1), (10, 1)], [False, False, False]),
        ([(3, 1), (5, 1), (10, 1)], [False, False, False]),
        ([(5, 10, 4), (5, 10, 4, 1)], [True, True]),
    ],
)
def test__not_time_series_after_snapshot(
    shapes: list[tuple[int, ...]], time_series: list[bool]
):
    tensors = [
        phlower_tensor(np.random.rand(*shape), is_time_series=ts)
        for shape, ts in zip(shapes, time_series, strict=True)
    ]
    collection = phlower_tensor_collection(
        {f"sample_{i}": t for i, t in enumerate(tensors)}
    )

    snapped_collectiton = collection.snapshot(0)
    assert snapped_collectiton.get_time_series_length() == 0


@pytest.mark.parametrize(
    "shapes, time_series",
    [
        ([(3, 10, 4), (10, 4)], [True, False]),
        ([(3, 5, 1), (5, 1)], [True, False]),
        ([(5, 1), (5, 1), (10, 1)], [False, False, False]),
        ([(3, 1), (5, 1), (10, 1)], [False, False, False]),
        ([(5, 10, 4), (5, 10, 4, 1)], [True, True]),
    ],
)
def test__snapshot_content(
    shapes: list[tuple[int, ...]], time_series: list[bool]
):
    tensors = [
        phlower_tensor(np.random.rand(*shape), is_time_series=ts)
        for shape, ts in zip(shapes, time_series, strict=True)
    ]
    collection = phlower_tensor_collection(
        {f"sample_{i}": t for i, t in enumerate(tensors)}
    )
    keys = list(collection.keys())

    for time in range(collection.get_time_series_length()):
        snapped_collectiton = collection.snapshot(time)

        for k in keys:
            assert k in snapped_collectiton
            assert snapped_collectiton[k].is_time_series is False
            if collection[k].is_time_series:
                np.testing.assert_array_almost_equal(
                    collection[k][time].numpy(), snapped_collectiton[k].numpy()
                )
            else:
                np.testing.assert_array_almost_equal(
                    collection[k].numpy(), snapped_collectiton[k].numpy()
                )


@pytest.mark.parametrize(
    "weights, desired_keys, desired_coeff",
    [
        (None, ["a", "b", "c"], 1.0),
        ({"a": 1.0, "b": 1.0, "c": 1.0}, ["a", "b", "c"], 1.0),
        ({"a": 1.0, "b": 2.0, "c": 1.0}, ["a", "b", "b", "c"], 1.0),
        ({"a": 0.0, "b": 1.0, "c": 0.0}, ["b"], 1.0),
        ({"a": 0.5, "b": 0.5, "c": 0.0}, ["a", "b"], 0.5),
    ],
)
def test__collections_sum_with_weights(
    weights: dict[str, float],
    desired_keys: list[str],
    desired_coeff: float,
):
    data = phlower_tensor_collection(
        {
            "a": torch.rand(1),
            "b": torch.rand(1),
            "c": torch.rand(1),
        }
    )

    actual = data.sum(weights=weights)

    if weights is None:
        weights = {k: 1.0 for k in data.keys()}
    desired = sum(data[k] for k in desired_keys) * desired_coeff
    np.testing.assert_array_almost_equal(
        actual.numpy(),
        desired.numpy(),
    )


@pytest.mark.parametrize(
    "weights",
    [
        {"a": 0.2, "b": 0.3, "c": 0.5},
        {"a": 1.0, "b": 0.0, "c": 0.0},
        {"a": 0.0, "b": 0.0, "c": 1.0},
    ],
)
def test__collections_same_value_mean_and_weight(
    weights: dict[str, float],
):
    data = phlower_tensor_collection(
        {
            "a": torch.rand(1),
            "b": torch.rand(1),
            "c": torch.rand(1),
        }
    )

    sum_actual = data.sum(weights=weights)
    mean_actual = data.mean(weights=weights)

    np.testing.assert_array_almost_equal(
        mean_actual.numpy(),
        sum_actual.numpy(),
    )


@pytest.mark.parametrize(
    "weights, desired_keys",
    [
        (None, ["a", "b", "c"]),
        ({"a": 1.0, "b": 1.0, "c": 1.0}, ["a", "b", "c"]),
        ({"a": 1.0, "b": 2.0, "c": 1.0}, ["a", "b", "b", "c"]),
        ({"a": 0.0, "b": 1.0, "c": 0.0}, ["b"]),
        ({"a": 0.5, "b": 0.5, "c": 0.0}, ["a", "b"]),
        ({"a": 2.0, "b": 3.0, "c": 1.0}, ["a", "a", "b", "b", "b", "c"]),
    ],
)
def test__collections_mean_with_weights(
    weights: dict[str, float], desired_keys: list[str]
):
    data = phlower_tensor_collection(
        {
            "a": torch.rand(1),
            "b": torch.rand(1),
            "c": torch.rand(1),
        }
    )

    actual = data.mean(weights=weights)

    if weights is None:
        weights = {k: 1.0 for k in data.keys()}
    desired = torch.mean(torch.stack([data[k] for k in desired_keys]))
    np.testing.assert_array_almost_equal(
        actual.numpy(),
        desired.numpy(),
    )
