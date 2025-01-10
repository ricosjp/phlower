from collections.abc import Callable
from typing import TypeVar

import hypothesis.strategies as st
import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import given
from phlower._base import PhysicalDimensions, phlower_array
from phlower._base.array.sparse import SparseArrayWrapper
from phlower.utils.enums import PhysicalDimensionSymbolType


@st.composite
def random_sparse_array(
    draw: Callable, arr_shape: st.SearchStrategy
) -> sp.coo_array:
    rng = np.random.default_rng()
    shapes = draw(arr_shape)
    if shapes[0] == 1 and shapes[1] == 1:
        return sp.random(*shapes, density=1, random_state=rng)

    else:
        return sp.random(*shapes, density=0.2, random_state=rng)


@st.composite
def random_physical_dimensions(
    draw: Callable,
) -> PhysicalDimensions:
    dimensions = draw(
        st.lists(
            elements=st.floats(allow_nan=False, allow_infinity=False, width=32),
            min_size=len(PhysicalDimensionSymbolType),
            max_size=len(PhysicalDimensionSymbolType),
        )
    )
    dimensions_dict = {
        name: dimensions[i]
        for i, name in enumerate(PhysicalDimensionSymbolType.__members__)
    }

    return PhysicalDimensions(dimensions_dict)


@given(
    random_sparse_array(
        st.lists(
            st.integers(min_value=1, max_value=100), min_size=2, max_size=2
        )
    )
)
def test__sparse_array_property(arr: sp.coo_array):
    phlower_arr = phlower_array(arr)

    assert isinstance(phlower_arr, SparseArrayWrapper)
    assert phlower_arr.is_sparse
    assert not phlower_arr.is_time_series
    assert not phlower_arr.is_voxel

    assert phlower_arr.shape == arr.shape

    np.testing.assert_array_almost_equal(phlower_arr.data, arr.data)

    np.testing.assert_array_almost_equal(
        phlower_arr.to_numpy().todense(), arr.todense()
    )

    with pytest.raises(ValueError) as ex:
        _ = phlower_arr.slice_along_time_axis(slice(1, 1, 1))

    assert "slice_along_time_axis is not allowed" in str(ex.value)


@given(
    arr=random_sparse_array(
        st.lists(
            st.integers(min_value=1, max_value=100), min_size=2, max_size=2
        )
    ),
    dimensions=random_physical_dimensions(),
)
def test__sparse_array_to_phlower_tensor(
    arr: sp.coo_array, dimensions: PhysicalDimensions
):
    phlower_arr = phlower_array(arr, dimensions=dimensions)
    phlower_tensor = phlower_arr.to_tensor()

    assert phlower_tensor.is_sparse


T = TypeVar("T")


def dummy_apply_function(x: T) -> T:
    return np.abs(x / np.max(x))


@given(
    random_sparse_array(
        st.lists(
            st.integers(min_value=1, max_value=100), min_size=2, max_size=2
        )
    )
)
def test__sparse_array_apply(arr: sp.coo_array):
    phlower_arr = phlower_array(arr)

    results = phlower_arr.apply(dummy_apply_function, componentwise=True)

    np.testing.assert_array_almost_equal(
        results.to_numpy().todense(), dummy_apply_function(arr).todense()
    )


@given(
    random_sparse_array(
        st.lists(
            st.integers(min_value=1, max_value=100), min_size=2, max_size=2
        )
    )
)
def test__sparse_array_apply_not_allowed_to_use_diagonal(arr: sp.coo_array):
    phlower_arr = phlower_array(arr)

    with pytest.raises(ValueError) as ex:
        _ = phlower_arr.apply(dummy_apply_function, True, use_diagonal=True)

    assert "Cannot set use_diagonal=True in self.apply function" in str(
        ex.value
    )
