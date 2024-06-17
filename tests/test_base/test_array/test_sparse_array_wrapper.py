import pytest

import scipy.sparse as sp
import numpy as np

from phlower._base.array.sparse import SparseArrayWrapper, concatenate, decompose


@pytest.mark.parametrize("shapes, expected_shape", [
    ([(5, 6), (4, 9), (10, 11)], (19, 26)),
    ([(1, 1), (2, 1), (1, 1)], (4, 3)),
    ([(3, 5)], (3, 5))
])
def test__concatenate(shapes, expected_shape):
    rng = np.random.default_rng()

    sparse_arrays = [
        SparseArrayWrapper(
            sp.random(*arr_shape, density=0.1, random_state=rng)
        )
        for arr_shape in shapes
    ]

    concat_arr = concatenate(sparse_arrays)

    assert concat_arr.shape == expected_shape
    assert concat_arr.batch_info.shapes == shapes


@pytest.mark.parametrize("shapes, expected_shape", [
    ([[(5, 6),  (3, 5)], [(4, 9)]], (12, 20))
])
def test__concatenate_nested_arrays(shapes: list[list[tuple[int]]], expected_shape):
    rng = np.random.default_rng()

    sparse_arrays: list[SparseArrayWrapper] = []
    concat_shapes: list[tuple[int]] = []
    for _shapes in shapes:
        _arrs = [
            SparseArrayWrapper(
                sp.random(*arr_shape, density=0.1, random_state=rng)
            )
            for arr_shape in _shapes
        ]
        concat_shapes += _shapes
        sparse_arrays.append(
            concatenate(_arrs)
        )

    concat_arr = concatenate(sparse_arrays)
    assert concat_arr.shape == expected_shape
    assert concat_arr.batch_info.shapes == list(concat_shapes)



@pytest.mark.parametrize("shapes", [
    ([(5, 6), (4, 9), (10, 11)]),
    ([(1, 1), (2, 1), (1, 1)]),
    ([(3, 5)])
])
def test__decompose(shapes):
    rng = np.random.default_rng()

    sparse_arrays = [
        SparseArrayWrapper(
            sp.random(*arr_shape, density=0.1, random_state=rng)
        )
        for arr_shape in shapes
    ]

    concat_arr = concatenate(sparse_arrays)
    results = decompose(concat_arr)

    assert len(results) == len(shapes)

    for i in range(len(results)):
        assert results[i].shape == shapes[i] 
        np.testing.assert_array_almost_equal(
            sparse_arrays[i].data, results[i].data
        ) 


@pytest.mark.parametrize("shapes", [
    ([(5, 6), (4, 9), (10, 11)]),
    ([(1, 1), (2, 1), (1, 1)]),
    ([(3, 5)])
])
def test__decompose_of_instance_method(shapes):
    rng = np.random.default_rng()

    sparse_arrays = [
        SparseArrayWrapper(
            sp.random(*arr_shape, density=0.1, random_state=rng)
        )
        for arr_shape in shapes
    ]

    concat_arr = concatenate(sparse_arrays)
    results = concat_arr.decompose()

    assert len(results) == len(shapes)

    for i in range(len(results)):
        assert results[i].shape == shapes[i] 
        np.testing.assert_array_almost_equal(
            sparse_arrays[i].data, results[i].data
        ) 

