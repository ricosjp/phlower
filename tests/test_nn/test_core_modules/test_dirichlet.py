import numpy as np
import pytest
import torch
from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import Dirichlet


def test__can_call_parameters():
    model = Dirichlet("identity")

    # To check Dirichlet inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "input_shapes, desired_shape",
    [
        ([(1, 2, 3), (1, 2, 3)], (1, 2, 3)),
        ([(2, 3, 4), (3, 4)], (2, 3, 4)),
    ],
)
def test__concatenated_tensor_shape(
    input_shapes: list[tuple[int]], desired_shape: tuple[int]
):
    phlower_tensors = {
        f"phlower_tensor_{i}": PhlowerTensor(
            torch.from_numpy(np.random.rand(*s))
        )
        for i, s in enumerate(input_shapes)
    }
    phlower_tensors["phlower_tensor_1"][
        tuple(0 for i in range(len(input_shapes[1])))
    ] = float("nan")
    phlower_tensors = phlower_tensor_collection(phlower_tensors)

    model = Dirichlet("identity")

    actual = model(phlower_tensors)

    assert actual.shape == desired_shape

    desired = np.copy(phlower_tensors["phlower_tensor_0"].to_numpy())
    if len(input_shapes[0]) == len(input_shapes[1]):
        desired = np.copy(phlower_tensors["phlower_tensor_1"].to_numpy())
        desired[0, 0, 0] = phlower_tensors["phlower_tensor_0"].to_numpy()[
            0, 0, 0
        ]
    elif len(input_shapes[0]) == len(input_shapes[1]) + 1:
        desired[:] = np.copy(phlower_tensors["phlower_tensor_1"].to_numpy())
        for i in range(input_shapes[0][0]):
            desired[i, 0, 0] = phlower_tensors["phlower_tensor_0"].to_numpy()[
                i, 0, 0
            ]
    else:
        raise ValueError("input_shapes value error")

    np.testing.assert_almost_equal(
        desired,
        actual.to_numpy(),
    )
