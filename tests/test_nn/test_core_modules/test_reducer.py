import numpy as np
import pytest
import torch
from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import Reducer


def test__can_call_parameters():
    model = Reducer("identity", operator="add")

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "input_shape, activation, num, operator, desired_shape",
    [
        ((5, 5, 16), "identity", 2, "add", (5, 5, 16)),
        ((4, 2, 16), "identity", 3, "add", (4, 2, 16)),
        ((5, 2, 3, 4), "tanh", 2, "mul", (5, 2, 3, 4)),
        ((3, 2, 1), "relu", 3, "mul", (3, 2, 1)),
    ],
)
def test__accessed_tensor_shape(
    input_shape: tuple[int],
    activation: str,
    num: int,
    operator: str,
    desired_shape: tuple[int],
):
    phlower_tensors = {
        f"phlower_tensor_{i}": PhlowerTensor(
            torch.from_numpy(np.random.rand(*input_shape))
        )
        for i in range(num)
    }
    phlower_tensors = phlower_tensor_collection(phlower_tensors)

    model = Reducer(activation=activation, operator=operator)

    actual: PhlowerTensor = model(phlower_tensors)

    assert actual.shape == desired_shape
