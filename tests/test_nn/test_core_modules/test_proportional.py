import numpy as np
import pytest
import torch

from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import Proportional


def test__can_call_parameters():
    model = Proportional(nodes=[4, 8])

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "size, is_time_series",
    [
        ((10, 1), False),
        ((10, 16), False),
        ((10, 3, 16), False),
        ((4, 10, 1), True),
        ((4, 10, 16), True),
        ((4, 10, 3, 16), True),
    ],
)
@pytest.mark.parametrize("n_output_feature", [1, 16, 32])
@pytest.mark.parametrize("scale", [0., 0.5, 2.])
def test__proportional_linearity(
        size, is_time_series, n_output_feature, scale):

    model = Proportional(nodes=[size[-1], n_output_feature])

    phlower_tensor = PhlowerTensor(
        torch.rand(*size), is_time_series=is_time_series)

    phlower_tensors = phlower_tensor_collection({'tensor': phlower_tensor})
    actual = model(phlower_tensors).to_numpy()

    scaled_phlower_tensors = phlower_tensor_collection(
        {'tensor': phlower_tensor * scale})
    desired = model(scaled_phlower_tensors).to_numpy()

    np.testing.assert_almost_equal(actual * scale, desired)
