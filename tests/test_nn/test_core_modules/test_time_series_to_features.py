import numpy as np
import pytest
import torch
from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import TimeSeriesToFeatures


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

    actual: PhlowerTensor = model(phlower_tensors)

    assert actual.shape == desired_shape

    if activation == "identity":
        s = len(input_shape)
        shape = (i for i in range(1, s))

        ans = np.reshape(
            np.transpose(phlower_tensors.to_numpy()["tensor"], (*shape, 0)),
            list(input_shape[1:-1]) + [-1],
        )
        act = actual.to_tensor().to("cpu").detach().numpy().copy()

        np.testing.assert_almost_equal(ans, act)
