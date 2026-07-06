import numpy as np
import pytest
import torch
from phlower_tensor import PhlowerTensor
from phlower_tensor.collections import phlower_tensor_collection

from phlower.nn import Activation


def test__can_call_parameters():
    model = Activation("identity")

    # To check Einsum inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize("shape", [(10, 16), (10, 3, 16)])
@pytest.mark.parametrize(
    "activation, torch_activation",
    [
        ("identity", lambda x: x),
        ("tanh", torch.tanh),
        ("relu", torch.relu),
    ],
)
def test__activation(
    shape: tuple[int], activation: str, torch_activation: callable
):
    ptc = phlower_tensor_collection(
        {"t": PhlowerTensor(torch.from_numpy(np.random.rand(*shape)))}
    )

    model = Activation(activation)

    actual = model(ptc)

    assert actual.shape == shape
    desired = torch_activation(ptc["t"].to_tensor())
    np.testing.assert_almost_equal(actual.numpy(), desired.numpy())
