import numpy as np
import pytest
import torch
from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import ActivationSelector, TimeSeriesToFeatures


def test__can_call_parameters():
    model = TimeSeriesToFeatures("identity")

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "input_shape, activation, desired_shape",
    [
        ((5, 3, 16), "identity", (3, 16 * 5)),
        ((4, 2, 16), "identity", (2, 16 * 4)),
        ((5, 2, 3, 4), "tanh", (2, 3, 4 * 5)),
        ((3, 2, 1), "relu", (2, 1 * 3)),
    ],
)
def test__accessed_tensor_shape(
    input_shape: tuple[int],
    activation: str,
    desired_shape: tuple[int],
):
    phlower_tensor = PhlowerTensor(
        torch.from_numpy(np.random.rand(*input_shape)), is_time_series=True
    )
    phlower_tensors = phlower_tensor_collection({"tensor": phlower_tensor})

    model = TimeSeriesToFeatures(activation=activation)
    actual: PhlowerTensor = model.forward(phlower_tensors)
    assert actual.shape == desired_shape

    activate_func = ActivationSelector.select(activation)

    shape = list(range(1, len(input_shape)))
    desired = activate_func(
        torch.reshape(
            torch.permute(phlower_tensor.to_tensor(), (*shape, 0)),
            list(input_shape[1:-1]) + [-1],
        )
    )
    act = actual.to_tensor()

    np.testing.assert_array_almost_equal(desired, act)
