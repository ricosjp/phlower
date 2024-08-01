import numpy as np
import pytest
import torch

from phlower._base import phlower_dimension_tensor, phlower_tensor
from phlower._base._functionals import is_same_dimensions, is_same_layout


def _create_random_tensors(shapes: list[tuple], is_sparse: list[bool]):
    assert len(shapes) == len(is_sparse)
    tensors = [torch.tensor(np.random.rand(*shape)) for shape in shapes]
    tensors = [
        v.to_sparse_coo() if is_sparse[i] else v for i, v in enumerate(tensors)
    ]
    return [phlower_tensor(v) for v in tensors]


def _create_random_tensors_with_dims(
    shapes: list[tuple], dimensions: list[dict[str, float]]
):
    assert len(shapes) == len(dimensions)
    tensors = [torch.tensor(np.random.rand(*shape)) for shape in shapes]
    _dimensions = [phlower_dimension_tensor(v) for v in dimensions]
    return [phlower_tensor(v, _dimensions[i]) for i, v in enumerate(tensors)]


@pytest.mark.parametrize(
    "shapes, is_sparse, desired",
    [
        ([(3, 6)], [True], True),
        ([(3, 6), (4, 5)], [True, True], True),
        ([(3, 6), (4, 5), (10, 5)], [True, True, False], False),
        ([(3, 6), (11, 5)], [False, False], True),
    ],
)
def test__is_same_layout(shapes, is_sparse, desired):
    tensors = _create_random_tensors(shapes, is_sparse)
    assert is_same_layout(tensors) == desired


@pytest.mark.parametrize(
    "shapes, dimensions, desired",
    [
        (
            [(2, 3)],
            [{"M": 3, "L": 2}],
            True,
        ),
        (
            [(3, 6), (4, 5)],
            [{"M": 3, "L": 2}, {"M": 3, "L": 2}],
            True,
        ),
        (
            [(3, 6), (4, 5), (10, 5)],
            [
                {"M": 3, "L": 2},
                {"M": 3, "L": 2},
                {"M": 3, "L": 2, "T": -2},
            ],
            False,
        ),
        (
            [(3, 6), (11, 5)],
            [
                {"M": 3, "L": 2},
                {"I": 3, "N": 2},
            ],
            False,
        ),
    ],
)
def test__is_same_dimensions(shapes, dimensions, desired):
    tensors = _create_random_tensors_with_dims(shapes, dimensions)
    assert is_same_dimensions(tensors) == desired
