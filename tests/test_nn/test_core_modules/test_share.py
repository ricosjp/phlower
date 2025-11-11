from unittest import mock

import numpy as np
import pytest
import torch
from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import MLP, Share
from phlower.nn._interface_module import IReadonlyReferenceGroup

# def test__can_call_parameters():
#     model = Share(reference_name="MLP0")
#     MLP0 = MLP(nodes=[10, 10])
#     model._reference = MLP0

#     # To check Concatenator inherit torch.nn.Module appropriately
#     _ = model.parameters()


@pytest.mark.parametrize(
    "mlp_nodes",
    [([10, 10]), ([20, 10, 100])],
)
def test__reference_same_object(mlp_nodes: list[int]):
    model = Share(reference_name="MLP0")
    MLP0 = MLP(nodes=mlp_nodes)
    model._reference = MLP0

    phlower_tensors = phlower_tensor_collection(
        {"sample_input": PhlowerTensor(tensor=torch.rand(3, mlp_nodes[0]))}
    )

    mlp_val = MLP0(phlower_tensors)
    model_val = model.forward(phlower_tensors)

    np.testing.assert_array_almost_equal(
        mlp_val.to_tensor().detach(), model_val.to_tensor().detach()
    )


@pytest.mark.parametrize("reference_name", ["MLP0", "GCP0"])
def test__search_reference_name(reference_name: str):
    model = Share(reference_name=reference_name)

    mocked = mock.MagicMock(IReadonlyReferenceGroup)
    model.resolve(parent=mocked)

    mocked.search_module.assert_called_once_with(reference_name)
