import numpy as np
import pytest
import torch
from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import Accessor


def test__can_call_parameters():
    model = Accessor("identity")

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "input_shape, activation, index, desired_shape",
    [
        ((5, 5, 16), "identity", 0, (5, 16)),
        ((4, 2, 16), "identity", -1, (2, 16)),
        ((5, 2, 3, 4), "tanh", 1, (2, 3, 4)),
        ((3, 2, 1), "relu", 2, (2, 1)),
    ],
)
def test__accessed_tensor_shape(
    input_shape: tuple[int],
    activation: str,
    index: int,
    desired_shape: tuple[int],
):
    phlower_tensor = PhlowerTensor(
        torch.from_numpy(np.random.rand(*input_shape))
    )
    phlower_tensors = phlower_tensor_collection({"tensor": phlower_tensor})

    model = Accessor(activation=activation, index=index)

    actual: PhlowerTensor = model(phlower_tensors)

    assert actual.shape == desired_shape

    if activation == "identity":
        np.testing.assert_almost_equal(
            phlower_tensors.to_numpy()["tensor"][index],
            actual.to("cpu").detach().numpy().copy(),
        )
