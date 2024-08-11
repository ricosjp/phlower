
import numpy as np
import pytest
import torch

from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import MLP, PInvMLP


def test__can_call_parameters():
    model = PInvMLP(reference_name="MLP0")
    MLP0 = MLP(nodes=[10, 10])
    model._reference = MLP0

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "mlp_nodes, activations, decimal",
    [
        ([10, 10], ["identity"], 4),
        ([10, 12], ["leaky_relu0p5"], 4),
        ([20, 40, 100], ["tanh", "identity"], 4),
        ([20, 20, 40, 100], ["tanh", "smooth_leaky_relu", "leaky_relu0p5"], 3),
    ],
)
def test__pinv_mlp(mlp_nodes, activations, decimal):
    MLP0 = MLP(nodes=mlp_nodes, activations=activations)

    model = PInvMLP(reference_name="MLP0")
    model._reference = MLP0
    model._initialize()

    t = PhlowerTensor(tensor=torch.rand(10, 3, mlp_nodes[0]))
    phlower_tensors = phlower_tensor_collection({"tensor": t})

    mlp_val = MLP0(phlower_tensors)
    pinv_val = model(phlower_tensor_collection({"tensor": mlp_val}))

    np.testing.assert_array_almost_equal(
        pinv_val.to_numpy(), t.to_numpy(), decimal=decimal)
