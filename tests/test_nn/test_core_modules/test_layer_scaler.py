import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given
from phlower_tensor import PhlowerTensor
from phlower_tensor.collections import (
    phlower_tensor_collection,
)

from phlower.nn import LayerScaler
from phlower.settings._module_settings import (
    LayerScalerSetting,
    LayerScalingMethod,
)


def test__can_call_parameters():
    model = LayerScaler(scaling_method="signed_log1p")

    # To check LayerNorm inherit torch.nn.Module appropriately
    _ = model.parameters()


@given(
    scaling_method=st.sampled_from([e.value for e in LayerScalingMethod]),
)
def test__can_pass_parameters_via_setting(
    scaling_method: LayerScalingMethod,
):

    setting = LayerScalerSetting(
        scaling_method=scaling_method,
    )
    model = LayerScaler.from_setting(setting)

    assert model._scaling_method == scaling_method


@pytest.mark.parametrize(
    "scaling_method, input_values",
    [
        (
            "signed_log1p",
            torch.tensor([-1e5, -1e4, -1.0, 0.0, 1.0, 1e4, 1e5]),
        ),
        (
            "asinh",
            torch.tensor([-1e5, -1e4, -1.0, 0.0, 1.0, 1e4, 1e5]),
        ),
    ],
)
def test__squash_value_range(
    scaling_method: LayerScalingMethod, input_values: torch.Tensor
):
    model = LayerScaler(scaling_method=scaling_method)

    inputs = phlower_tensor_collection({"sample": input_values})
    output_tensor: PhlowerTensor = model(inputs)

    max_v = output_tensor.numpy().max()
    min_v = output_tensor.numpy().min()

    assert max_v < 20.0, f"max value {max_v} is not less than 10.0"
    assert min_v > -20.0, f"min value {min_v} is not greater than -10.0"
    print(f"scaling_method: {scaling_method}, max: {max_v}, min: {min_v}")


@pytest.mark.parametrize(
    "scaling_method, input_values",
    [
        (
            "signed_log1p",
            torch.tensor([-1e5, -1e4, -1.0, 0.0, 1.0, 1e4, 1e5]),
        ),
        (
            "asinh",
            torch.tensor([-1e5, -1e4, -1.0, 0.0, 1.0, 1e4, 1e5]),
        ),
    ],
)
def test__restore_sign(
    scaling_method: LayerScalingMethod, input_values: torch.Tensor
):
    model = LayerScaler(scaling_method=scaling_method)

    inputs = phlower_tensor_collection({"sample": input_values})
    output_tensor: PhlowerTensor = model(inputs)

    np.testing.assert_array_almost_equal(
        torch.sign(output_tensor.to_tensor()).numpy(),
        torch.sign(input_values).numpy(),
    )
