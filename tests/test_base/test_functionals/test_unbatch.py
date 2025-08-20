from collections.abc import Callable

import numpy as np
import pytest
import torch
from phlower._base._functionals import to_batch, unbatch
from phlower._base.tensors._interface import IPhlowerTensor
from phlower._fields import SimulationField
from phlower.collections.tensors import (
    phlower_tensor_collection,
)


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


@pytest.mark.parametrize(
    "name_to_shapes, concat_dim, name_to_dimension, "
    "is_time_series, name_to_expected_shape",
    [
        (
            {
                "t1": [(3, 5)],
            },
            0,
            {
                "t1": None,
            },
            False,
            {
                "t1": (3, 5),
            },
        ),
        (
            {
                "t1": [(3, 5), (4, 5), (10, 5)],
            },
            0,
            {
                "t1": None,
            },
            False,
            {
                "t1": (17, 5),
            },
        ),
        (
            {
                "t1": [(3, 5), (4, 5), (10, 5)],
                "t2": [(3, 3, 8), (4, 3, 8), (10, 3, 8)],
                "global": [(1, 2), (1, 2), (1, 2)],
            },
            0,
            {
                "t1": None,
                "t2": None,
                "global": None,
            },
            False,
            {
                "t1": (17, 5),
                "t2": (17, 3, 8),
                "global": (3, 2),
            },
        ),
        (
            {
                "t1": [(5, 3, 1), (5, 4, 1), (5, 10, 1)],
                "t2": [(5, 3, 3, 8), (5, 4, 3, 8), (5, 10, 3, 8)],
                "global": [(5, 1, 2), (5, 1, 2), (5, 1, 2)],
            },
            1,
            {
                "t1": None,
                "t2": None,
                "global": None,
            },
            True,
            {
                "t1": (5, 17, 1),
                "t2": (5, 17, 3, 8),
                "global": (5, 3, 2),
            },
        ),  # time-series
        (
            {
                "t1": [(6, 2), (5, 2), (11, 2)],
                "t2": [(6, 3, 8), (5, 3, 8), (11, 3, 8)],
                "global": [(1, 2), (1, 2), (1, 2)],
            },
            0,
            {
                "t1": [
                    {"T": 3, "I": -1, "J": 2},
                    {"T": 3, "I": -1, "J": 2},
                    {"T": 3, "I": -1, "J": 2},
                ],
                "t2": [
                    {"T": 1, "I": 1, "J": 2},
                    {"T": 1, "I": 1, "J": 2},
                    {"T": 1, "I": 1, "J": 2},
                ],
                "global": [
                    {"T": 2, "I": -1, "J": 2},
                    {"T": 2, "I": -1, "J": 2},
                    {"T": 2, "I": -1, "J": 2},
                ],
            },
            False,
            {
                "t1": (22, 2),
                "t2": (22, 3, 8),
                "global": (3, 2),
            },
        ),
    ],
)
def test__unbatch_dict(
    name_to_shapes: dict[str, list[tuple[int]]],
    concat_dim: int,
    name_to_dimension: dict[str, dict | None],
    is_time_series: bool,
    name_to_expected_shape: dict[str, tuple[int]],
    create_dense_tensors: Callable[
        [list[tuple[int]], list[dict[str, float]] | None, bool],
        list[IPhlowerTensor],
    ],
):
    dict_tensors = {
        k: create_dense_tensors(
            name_to_shapes[k], name_to_dimension[k], is_time_series
        )
        for k in name_to_shapes.keys()
    }
    dict_batched_tensor = {
        k: to_batch(v, concat_dim)[0] for k, v in dict_tensors.items()
    }
    batch_info = [to_batch(v, concat_dim)[1] for v in dict_tensors.values()][0]
    for k, v in dict_batched_tensor.items():
        assert v.shape == name_to_expected_shape[k]

    concat_tensor_collection = phlower_tensor_collection(dict_batched_tensor)
    unbatched_tensor_collection = unbatch(
        concat_tensor_collection,
        n_nodes=batch_info.n_nodes,
    )

    for key, shapes in name_to_shapes.items():
        unbatched_tensors = [d[key] for d in unbatched_tensor_collection]
        assert len(unbatched_tensors) == len(shapes)

        for i in range(len(shapes)):
            np.testing.assert_array_almost_equal(
                unbatched_tensors[i].to_tensor(),
                dict_tensors[key][i].to_tensor(),
            )

        dimension = name_to_dimension[key]
        if dimension is not None:
            for i in range(len(shapes)):
                assert (
                    unbatched_tensors[i].dimension
                    == dict_tensors[key][i].dimension
                )


@pytest.mark.parametrize(
    "name_to_shapes, concat_dim, name_to_dimension, "
    "is_time_series, name_to_expected_shape",
    [
        (
            {
                "t1": [(3, 5)],
            },
            0,
            {
                "t1": None,
            },
            False,
            {
                "t1": (3, 5),
            },
        ),
        (
            {
                "t1": [(3, 5), (4, 5), (10, 5)],
            },
            0,
            {
                "t1": None,
            },
            False,
            {
                "t1": (17, 5),
            },
        ),
        (
            {
                "t1": [(3, 5), (4, 5), (10, 5)],
                "t2": [(3, 3, 8), (4, 3, 8), (10, 3, 8)],
                "global": [(1, 2), (1, 2), (1, 2)],
            },
            0,
            {
                "t1": None,
                "t2": None,
                "global": None,
            },
            False,
            {
                "t1": (17, 5),
                "t2": (17, 3, 8),
                "global": (3, 2),
            },
        ),
        (
            {
                "t1": [(5, 3, 1), (5, 4, 1), (5, 10, 1)],
                "t2": [(5, 3, 3, 8), (5, 4, 3, 8), (5, 10, 3, 8)],
                "global": [(5, 1, 2), (5, 1, 2), (5, 1, 2)],
            },
            1,
            {
                "t1": None,
                "t2": None,
                "global": None,
            },
            True,
            {
                "t1": (5, 17, 1),
                "t2": (5, 17, 3, 8),
                "global": (5, 3, 2),
            },
        ),  # time-series
        (
            {
                "t1": [(6, 2), (5, 2), (11, 2)],
                "t2": [(6, 3, 8), (5, 3, 8), (11, 3, 8)],
                "global": [(1, 2), (1, 2), (1, 2)],
            },
            0,
            {
                "t1": [
                    {"T": 3, "I": -1, "J": 2},
                    {"T": 3, "I": -1, "J": 2},
                    {"T": 3, "I": -1, "J": 2},
                ],
                "t2": [
                    {"T": 1, "I": 1, "J": 2},
                    {"T": 1, "I": 1, "J": 2},
                    {"T": 1, "I": 1, "J": 2},
                ],
                "global": [
                    {"T": 2, "I": -1, "J": 2},
                    {"T": 2, "I": -1, "J": 2},
                    {"T": 2, "I": -1, "J": 2},
                ],
            },
            False,
            {
                "t1": (22, 2),
                "t2": (22, 3, 8),
                "global": (3, 2),
            },
        ),
    ],
)
def test__unbatch_field(
    name_to_shapes: dict[str, list[tuple[int]]],
    concat_dim: int,
    name_to_dimension: dict[str, dict | None],
    is_time_series: bool,
    name_to_expected_shape: dict[str, tuple[int]],
    create_dense_tensors: Callable[
        [list[tuple[int]], list[dict[str, float]] | None, bool],
        list[IPhlowerTensor],
    ],
):
    dict_tensors = {
        k: create_dense_tensors(
            name_to_shapes[k], name_to_dimension[k], is_time_series
        )
        for k in name_to_shapes.keys()
    }
    dict_batched_tensor = {
        k: to_batch(v, concat_dim)[0] for k, v in dict_tensors.items()
    }
    batch_info = [to_batch(v, concat_dim)[1] for v in dict_tensors.values()][0]
    for k, v in dict_batched_tensor.items():
        assert v.shape == name_to_expected_shape[k]

    concat_tensor_collection = phlower_tensor_collection(dict_batched_tensor)

    concat_tensor_field = SimulationField(
        concat_tensor_collection,
        batch_info=batch_info,
    )

    unbatched_tensor_field = unbatch(
        concat_tensor_field,
        n_nodes=batch_info.n_nodes,
    )

    for key, shapes in name_to_shapes.items():
        unbatched_tensors = [d[key] for d in unbatched_tensor_field]
        assert len(unbatched_tensors) == len(shapes)

        for i in range(len(shapes)):
            np.testing.assert_array_almost_equal(
                unbatched_tensors[i].to_tensor(),
                dict_tensors[key][i].to_tensor(),
            )

        dimension = name_to_dimension[key]
        if dimension is not None:
            for i in range(len(shapes)):
                assert (
                    unbatched_tensors[i].dimension
                    == dict_tensors[key][i].dimension
                )
