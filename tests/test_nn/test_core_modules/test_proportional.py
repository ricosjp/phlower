import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import Proportional
from phlower.settings._module_settings import ProportionalSetting


def test__can_call_parameters():
    model = Proportional(nodes=[4, 8])

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@given(
    nodes=st.lists(
        st.integers(min_value=1, max_value=10), min_size=2, max_size=5
    ),
)
def test__can_pass_parameters_via_setting(
    nodes: list[int],
):
    setting = ProportionalSetting(
        nodes=nodes,
    )

    model = Proportional.from_setting(setting)

    assert model._nodes == nodes


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
@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0])
def test__proportional_linearity(
    size: tuple[int], is_time_series: bool, n_output_feature: int, scale: float
):
    model = Proportional(nodes=[size[-1], n_output_feature])

    phlower_tensor = PhlowerTensor(
        torch.rand(*size), is_time_series=is_time_series
    )

    phlower_tensors = phlower_tensor_collection({"tensor": phlower_tensor})
    actual = model(phlower_tensors).to_numpy()

    scaled_phlower_tensors = phlower_tensor_collection(
        {"tensor": phlower_tensor * scale}
    )
    desired = model(scaled_phlower_tensors).to_numpy()

    np.testing.assert_almost_equal(actual * scale, desired)
