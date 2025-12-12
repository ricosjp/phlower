from functools import reduce

import numpy as np
import pytest
import torch
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import phlower_tensor_collection

from phlower.nn import Reducer


def test__can_call_parameters():
    model = Reducer("identity", operator="add")

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "input_shape, activation, num, operator, desired_operator, desired_shape",
    [
        ((5, 5, 16), "relu", 2, "add", torch.add, (5, 5, 16)),
        ((4, 2, 16), "identity", 3, "add", torch.add, (4, 2, 16)),
        ((5, 2, 3, 4), "relu", 2, "mul", torch.multiply, (5, 2, 3, 4)),
        ((3, 2, 1), "identity", 3, "mul", torch.multiply, (3, 2, 1)),
    ],
)
def test__accessed_tensor_shape(
    input_shape: tuple[int],
    activation: str,
    num: int,
    operator: str,
    desired_operator: type[torch.add],
    desired_shape: tuple[int],
):
    np_tensors = [np.random.rand(*input_shape) for i in range(num)]
    phlower_tensors = {
        f"phlower_tensor_{i}": phlower_tensor(torch.from_numpy(np_tensors[i]))
        for i in range(num)
    }
    phlower_tensors = phlower_tensor_collection(phlower_tensors)

    model = Reducer(activation=activation, operator=operator)

    actual = model.forward(phlower_tensors)

    assert actual.shape == desired_shape

    ans = phlower_tensors["phlower_tensor_0"]
    for i in range(1, num):
        ans = desired_operator(ans, phlower_tensors[f"phlower_tensor_{i}"])

    np.testing.assert_array_almost_equal(
        ans,
        actual.to_tensor(),
    )


@pytest.mark.parametrize(
    "input_shapes, input_time_series, input_voxel, desired_shape",
    [
        ([(10, 3, 1), (10, 1), (10, 1, 1)], False, False, (10, 3, 1)),
        ([(5, 3, 1), (5,), (5, 1)], False, False, (5, 3, 1)),
        (
            [
                (2, 5, 3, 1),
                (
                    1,
                    5,
                ),
                (2, 5, 1),
            ],
            True,
            False,
            (2, 5, 3, 1),
        ),
        (
            [
                (1, 5, 3, 1),
                (
                    1,
                    5,
                ),
                (3, 5, 1),
            ],
            True,
            False,
            (3, 5, 3, 1),
        ),
        (
            [(1, 2, 3, 1), (1, 2, 3, 5, 1), (1, 2, 3, 1)],
            False,
            True,
            (1, 2, 3, 5, 1),
        ),
        (
            [(4, 1, 2, 3, 1), (1, 1, 2, 3, 5, 1), (1, 1, 2, 3, 1)],
            True,
            True,
            (4, 1, 2, 3, 5, 1),
        ),
    ],
)
def test__can_broadcast_with_different_size_shape(
    input_shapes: list[tuple[int]],
    input_time_series: bool,
    input_voxel: bool,
    desired_shape: tuple[int],
):
    phlower_tensors = {
        f"phlower_tensor_{i}": phlower_tensor(
            torch.rand(*shape),
            is_time_series=input_time_series,
            is_voxel=input_voxel,
        )
        for i, shape in enumerate(input_shapes)
    }
    phlower_tensors = phlower_tensor_collection(phlower_tensors)

    model = Reducer(activation="identity", operator="add")

    actual = model.forward(phlower_tensors)

    assert actual.shape == desired_shape


@pytest.mark.parametrize(
    "input_shapes, input_time_series, input_voxel",
    [
        ([(2, 2), (2, 1)], False, False),
        ([(5, 3, 1), (5, 1, 1)], False, False),
        ([(10, 3, 1), (10, 1, 1), (10, 3, 1)], False, False),
        ([(2, 10, 3, 1), (1, 10, 1, 1), (1, 10, 3, 1)], True, False),
        ([(1, 2, 3, 4, 5), (1, 2, 3, 4, 1)], False, True),
        ([(5, 1, 2, 3, 4, 5), (5, 1, 2, 3, 1, 1)], True, True),
    ],
)
def test__same_broadcast_with_torch(
    input_shapes: list[tuple[int]], input_time_series: bool, input_voxel: bool
):
    phlower_tensors = {
        f"phlower_tensor_{i}": phlower_tensor(
            torch.rand(*shape),
            is_voxel=input_voxel,
            is_time_series=input_time_series,
        )
        for i, shape in enumerate(input_shapes)
    }
    phlower_tensors = phlower_tensor_collection(phlower_tensors)

    model = Reducer(activation="identity", operator="add")
    actual = model.forward(phlower_tensors)

    desired = reduce(
        lambda x, y: torch.add(x, y),
        [
            phlower_tensors[f"phlower_tensor_{i}"].to_tensor()
            for i in range(len(input_shapes))
        ],
    )

    np.testing.assert_array_almost_equal(
        actual.to_tensor(),
        desired,
    )
