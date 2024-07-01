import numpy as np
import pytest

from phlower._base._functionals import to_batch, unbatch


@pytest.mark.parametrize(
    "shapes, expected_shape",
    [
        ([(5, 6), (4, 9), (10, 11)], (19, 26)),
        ([(1, 1), (2, 1), (1, 1)], (4, 3)),
        ([(3, 5)], (3, 5)),
    ],
)
def test__unbatch(shapes, expected_shape, create_sparse_tensors):
    sparse_tensors = create_sparse_tensors(shapes)

    concat_tensor, batch_info = to_batch(sparse_tensors)
    assert concat_tensor.shape == expected_shape

    tensors = unbatch(concat_tensor, batch_info)
    assert len(tensors) == len(shapes)

    for i in range(len(shapes)):
        np.testing.assert_array_almost_equal(
            sparse_tensors[i].to_tensor().to_dense(),
            tensors[i].to_tensor().to_dense(),
        )


# HACK: add tests for dense, sparse with dimensions
