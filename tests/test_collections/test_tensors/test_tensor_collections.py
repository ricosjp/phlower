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
