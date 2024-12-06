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
        ([(1, 1, 2), (1, 1, 2)], (1, 1, 2)),
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
    phlower_tensors["phlower_tensor_1"][0, 0, 0] = float("nan")
    phlower_tensors = phlower_tensor_collection(phlower_tensors)

    model = Dirichlet("identity")

    actual = model(phlower_tensors)

    assert actual.shape == desired_shape

    print(phlower_tensors["phlower_tensor_0"])
    print(phlower_tensors["phlower_tensor_1"])
    print(actual)

    desired = phlower_tensors["phlower_tensor_1"].to_numpy()
    desired[0, 0, 0] = phlower_tensors["phlower_tensor_0"].to_numpy()[0, 0, 0]

    np.testing.assert_almost_equal(
        desired,
        actual.to_numpy(),
    )
