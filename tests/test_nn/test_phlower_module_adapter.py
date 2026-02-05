import numpy as np
import pytest
import torch
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import (
    phlower_tensor_collection,
)

from phlower.nn._phlower_module_adapter import PhlowerModuleAdapter
from phlower.settings import ModuleSetting
from phlower.settings._debug_parameter_setting import (
    PhlowerModuleDebugParameters,
)
from phlower.utils.exceptions import PhlowerRunTimeError


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


@pytest.mark.parametrize(
    "output_tensor_shape", [(-1, 1), (2, 3, 2), (2, 3, 4, -1)]
)
def test__raise_error_invalid_output_tensor_shape(
    output_tensor_shape: list[int],
):
    debug_parameters = PhlowerModuleDebugParameters(
        output_tensor_shape=output_tensor_shape
    )
    setting = ModuleSetting(
        nn_type="Identity",
        name="aa",
        input_keys=["sample"],
        debug_parameters=debug_parameters,
    )
    input_tensor = phlower_tensor(torch.rand(2, 3, 4))
    input_tensors = phlower_tensor_collection({"sample": input_tensor})
    model = PhlowerModuleAdapter.from_setting(setting)
    with pytest.raises(PhlowerRunTimeError) as ex:
        _ = model.forward(input_tensors)

    assert "is different from desired shape" in str(ex.value)


@pytest.mark.parametrize(
    "output_tensor_shape", [(-1, -1, 4), (2, 3, 4), (2, -1, 4)]
)
def test__pass_output_tensor_shape(output_tensor_shape: list[int]):
    debug_parameters = PhlowerModuleDebugParameters(
        output_tensor_shape=output_tensor_shape
    )
    setting = ModuleSetting(
        nn_type="Identity",
        name="aa",
        input_keys=["sample"],
        debug_parameters=debug_parameters,
    )
    input_tensor = phlower_tensor(torch.rand(2, 3, 4))
    input_tensors = phlower_tensor_collection({"sample": input_tensor})
    model = PhlowerModuleAdapter.from_setting(setting)
    _ = model.forward(input_tensors)
