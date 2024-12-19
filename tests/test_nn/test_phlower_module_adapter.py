import numpy as np
import pytest
import torch
from phlower import phlower_tensor
from phlower.collections import phlower_tensor_collection
from phlower.nn._phlower_module_adapter import PhlowerModuleAdapter
from phlower.settings import ModuleSetting


@pytest.mark.parametrize("coeff", [1.0, 2.0, -3.2])
def test__coeff_factor_with_identity_module(coeff: float):
    setting = ModuleSetting(
        nn_type="Identity", name="aa", input_keys=["sample"], coeff=coeff
    )
    sample_input = phlower_tensor(torch.rand(2, 3))
    inputs = phlower_tensor_collection({"sample": sample_input})

    model = PhlowerModuleAdapter.from_setting(setting)
    actual = model.forward(inputs).unique_item()

    np.testing.assert_array_almost_equal(
        actual.to_tensor(), sample_input.to_tensor() * coeff
    )
