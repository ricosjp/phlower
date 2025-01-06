import numpy as np
import pytest
import scipy.sparse as sp
from phlower._base.array.sparse import SparseArrayWrapper
from phlower._base.array.sparse._sparse_array_wrapper import (
    _sparse_array_batch,
    _sparse_array_unbatch,
)


@pytest.mark.parametrize(
    "shapes, expected_shape",
    [
        ([(5, 6), (4, 9), (10, 11)], (19, 26)),
        ([(1, 1), (2, 1), (1, 1)], (4, 3)),
        ([(3, 5)], (3, 5)),
    ],
)
def test__batch(shapes: tuple[int], expected_shape: tuple[int]):
    rng = np.random.default_rng()

    sparse_arrays = [
        SparseArrayWrapper(sp.random(*arr_shape, density=0.1, random_state=rng))
        for arr_shape in shapes
    ]

    concat_arr, batch_info = _sparse_array_batch(sparse_arrays)

    assert concat_arr.shape == expected_shape
    assert batch_info.shapes == shapes


@pytest.mark.parametrize(
    "shapes",
    [([(5, 6), (4, 9), (10, 11)]), ([(1, 1), (2, 1), (1, 1)]), ([(3, 5)])],
)
def test__unbatch(shapes: list[tuple[int]]):
    rng = np.random.default_rng()

    sparse_arrays = [
        SparseArrayWrapper(sp.random(*arr_shape, density=0.1, random_state=rng))
        for arr_shape in shapes
    ]

    concat_arr, batch_info = _sparse_array_batch(sparse_arrays)
    results = _sparse_array_unbatch(concat_arr, batch_info)

    assert len(results) == len(shapes)

    for i in range(len(results)):
        assert results[i].shape == shapes[i]
        np.testing.assert_array_almost_equal(
            sparse_arrays[i].data, results[i].data
        )
