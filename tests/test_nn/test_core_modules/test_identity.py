import numpy as np
import pytest
import torch

from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import Identity


def test__can_call_parameters():
    model = Identity()

    # To check Identity inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "input_shape",
    [
        (5, 5, 32),
        (1, 2, 48),
    ],
)
def test__identity(input_shape):
    phlower_tensor = PhlowerTensor(torch.rand(*input_shape))
    phlower_tensors = phlower_tensor_collection(
        {"phlower_tensor": phlower_tensor}
    )

    model = Identity()

    actual = model(phlower_tensors)

    np.testing.assert_almost_equal(actual.to_numpy(), phlower_tensor.to_numpy())
