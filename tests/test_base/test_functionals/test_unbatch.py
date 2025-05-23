from collections.abc import Callable

import numpy as np
import pytest
import torch
from phlower._base._functionals import to_batch, unbatch
from phlower._base.tensors._interface import IPhlowerTensor


@pytest.mark.parametrize(
    "shapes, dimensions, expected_shape",
    [
        ([(5, 6), (4, 9), (10, 11)], None, (19, 26)),
        ([(1, 1), (2, 1), (1, 1)], None, (4, 3)),
        ([(3, 5)], None, (3, 5)),
        (
            [(5, 6), (4, 9), (10, 11)],
            [
                {"L": 2, "T": -2},
                {"L": 2, "T": -2},
                {"L": 2, "T": -2},
            ],
            (19, 26),
        ),
    ],
)
def test__unbatch_for_sparse(
    shapes: list[tuple[int]],
    dimensions: list[dict],
    expected_shape: tuple,
    create_sparse_tensors: Callable[
        [list[tuple[int]], list[dict[str, float]] | None], list[IPhlowerTensor]
    ],
):
    sparse_tensors = create_sparse_tensors(shapes, dimensions)

    concat_tensor, batch_info = to_batch(sparse_tensors)
    assert concat_tensor.shape == expected_shape

    tensors = unbatch(concat_tensor, batch_info)
    assert len(tensors) == len(shapes)

    for i in range(len(shapes)):
        np.testing.assert_array_almost_equal(
            sparse_tensors[i].to_tensor().to_dense(),
            tensors[i].to_tensor().to_dense(),
        )

    if dimensions is not None:
        for i in range(len(shapes)):
            assert tensors[i].dimension == sparse_tensors[i].dimension


@pytest.mark.parametrize(
    "shapes, concat_dim, dimensions, is_time_series, expected_shape",
    [
        ([(3, 5), (4, 5), (10, 5)], 0, None, False, (17, 5)),
        (
            [(5, 3, 1), (5, 4, 1), (5, 10, 1)],
            1,
            None,
            True,
            (5, 17, 1),
        ),  # time-series
        (
            [(6, 2), (5, 2), (11, 2)],
            0,
            [
                {"T": 3, "I": -1, "J": 2},
                {"T": 3, "I": -1, "J": 2},
                {"T": 3, "I": -1, "J": 2},
            ],
            False,
            (22, 2),
        ),
    ],
)
def test__unbatch_for_dense(
    shapes: list[tuple[int]],
    concat_dim: int,
    dimensions: dict | None,
    is_time_series: bool,
    expected_shape: tuple[int],
    create_dense_tensors: Callable[
        [list[tuple[int]], list[dict[str, float]] | None, bool],
        list[IPhlowerTensor],
    ],
):
    dense_tensors = create_dense_tensors(shapes, dimensions, is_time_series)

    concat_tensor, batch_info = to_batch(dense_tensors, concat_dim)
    assert concat_tensor.shape == expected_shape

    tensors = unbatch(concat_tensor, batch_info)
    assert len(tensors) == len(shapes)

    for i in range(len(shapes)):
        np.testing.assert_array_almost_equal(
            dense_tensors[i].to_tensor(),
            tensors[i].to_tensor(),
        )

    if dimensions is not None:
        for i in range(len(shapes)):
            assert tensors[i].dimension == dense_tensors[i].dimension


@pytest.mark.parametrize(
    "sparse_shapes, dense_shapes, expected_shape",
    [
        ([(5, 5), (4, 4), (10, 10)], [(5, 3), (4, 3), (10, 3)], (19, 3)),
        ([(1, 1), (2, 2), (1, 1)], [(1, 3), (2, 3), (1, 3)], (4, 3)),
        ([(10, 10)], [(10, 5)], (10, 5)),
    ],
)
def test__batched_tensor_mm(
    sparse_shapes: list[tuple[int]],
    dense_shapes: list[tuple[int]],
    expected_shape: tuple[int],
    create_sparse_tensors: Callable[
        [list[tuple[int]], list[dict[str, float]] | None], list[IPhlowerTensor]
    ],
    create_dense_tensors: Callable[
        [list[tuple[int]], list[dict[str, float]] | None], list[IPhlowerTensor]
    ],
):
    sparse_tensors = create_sparse_tensors(sparse_shapes)
    dense_tensors = create_dense_tensors(dense_shapes)

    sparse_ph, batch_info = to_batch(sparse_tensors)
    dense_ph, _ = to_batch(dense_tensors)
    mm_ph = torch.sparse.mm(sparse_ph, dense_ph)

    assert mm_ph.shape == expected_shape

    spmm_tensors = unbatch(mm_ph, batch_info)

    for i in range(len(sparse_shapes)):
        _desired = torch.sparse.mm(sparse_tensors[i], dense_tensors[i])
        np.testing.assert_array_almost_equal(
            spmm_tensors[i].to_tensor().numpy(), _desired.to_tensor().numpy()
        )
