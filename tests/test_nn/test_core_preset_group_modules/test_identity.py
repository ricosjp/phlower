import numpy as np
import pytest
import torch
from phlower_tensor import PhlowerTensor
from phlower_tensor.collections import phlower_tensor_collection

from phlower.nn._core_preset_group_modules import IdentityPresetGroupModule


def test__can_call_parameters():
    model = IdentityPresetGroupModule(
        input_to_output_map={},
    )

    _ = model.parameters()


def test__reference_name():
    model = IdentityPresetGroupModule(
        input_to_output_map={},
    )
    assert model.get_reference_name() is None


@pytest.mark.parametrize(
    "input_shape1, input_shape2",
    [
        (
            (2, 3, 4),
            (3, 4),
        ),
        (
            (2, 2),
            (4, 5),
        ),
    ],
)
def test__identity(input_shape1: tuple[int], input_shape2: tuple[int]):
    phlower_tensor1 = PhlowerTensor(torch.rand(*input_shape1))
    phlower_tensor2 = PhlowerTensor(torch.rand(*input_shape2))
    phlower_tensors = phlower_tensor_collection(
        {
            "phlower_tensor1": phlower_tensor1,
            "phlower_tensor2": phlower_tensor2,
        }
    )

    model = IdentityPresetGroupModule(
        input_to_output_map={
            "phlower_tensor1": "output1",
            "phlower_tensor2": "output2",
        }
    )

    actual = model.forward(phlower_tensors)

    np.testing.assert_almost_equal(
        actual["output1"].to_numpy(), phlower_tensor1.to_numpy()
    )
    np.testing.assert_almost_equal(
        actual["output2"].to_numpy(), phlower_tensor2.to_numpy()
    )
