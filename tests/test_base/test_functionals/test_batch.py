import pytest

from phlower._base._functionals import to_batch
from phlower._base.tensors import phlower_dimension_tensor


@pytest.mark.parametrize(
    "shapes, dimensions, desired_shape",
    [
        ([(3, 5), (4, 7), (10, 1)], None, (17, 13)),
        (
            [(3, 5), (4, 7), (10, 1)],
            [
                {"mass": 3, "length": 2},
                {"mass": 3, "length": 2},
                {"mass": 3, "length": 2},
            ],
            (17, 13),
        ),
    ],
)
def test__to_batch_for_sparse_tensors(
    shapes, dimensions, desired_shape, create_sparse_tensors
):
    tensors = create_sparse_tensors(shapes, dimensions)
    batched_tensor, batch_info = to_batch(tensors)

    assert batched_tensor.is_sparse
    if dimensions is not None:
        assert batched_tensor.dimension == phlower_dimension_tensor(
            dimensions[0]
        )
    else:
        assert batched_tensor.dimension is None
    assert batched_tensor.shape == desired_shape
    assert batch_info.shapes == shapes


@pytest.mark.parametrize(
    "shapes, concat_dim, dimensions, desired_shape",
    [
        ([(3, 5), (4, 5), (10, 5)], 0, None, (17, 5)),
        ([(5, 3), (5, 4), (5, 10)], 1, None, (5, 17)),
        (
            [(6, 2), (5, 2), (11, 2)],
            0,
            [
                {"time": 3, "electric_current": -1, "luminous_intensity": 2},
                {"time": 3, "electric_current": -1, "luminous_intensity": 2},
                {"time": 3, "electric_current": -1, "luminous_intensity": 2},
            ],
            (22, 2),
        ),
    ],
)
def test__to_batch_for_dense_tensors(
    shapes, concat_dim, dimensions, desired_shape, create_dense_tensors
):
    tensors = create_dense_tensors(shapes, dimensions)
    batched_tensor, batch_info = to_batch(tensors, concat_dim)

    assert not batched_tensor.is_sparse

    if dimensions is not None:
        assert batched_tensor.dimension == phlower_dimension_tensor(
            dimensions[0]
        )
    else:
        assert batched_tensor.dimension is None
    assert batched_tensor.shape == desired_shape
    assert batch_info.shapes == shapes
