import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given
from phlower_tensor import PhlowerTensor
from phlower_tensor.collections import phlower_tensor_collection

from phlower.nn import NaNToNum
from phlower.settings._module_settings import NaNToNumSetting


def test__can_call_parameters():
    model = NaNToNum(nodes=[10, 10])

    # To check LayerNorm inherit torch.nn.Module appropriately
    _ = model.parameters()


@given(
    nan=st.floats(-1e6, 1e6),
    posinf=st.one_of(st.floats(1e3, 1e6), st.none()),
    neginf=st.one_of(st.floats(-1e6, -1e3), st.none()),
)
def test__can_pass_parameters_via_setting(
    nan: float,
    posinf: float | None,
    neginf: float | None,
):
    setting = NaNToNumSetting(
        nodes=[10, 10],
        nan=nan,
        posinf=posinf,
        neginf=neginf,
    )
    model = NaNToNum.from_setting(setting)

    assert model._nan == nan
    assert model._posinf == posinf
    assert model._neginf == neginf


@pytest.mark.parametrize(
    "nan, posinf, neginf, inputs, expected",
    [
        (
            0.0,
            None,
            None,
            [float("nan"), 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ),
        (
            1.0,
            1e6,
            -1e6,
            [float("-inf"), -1.0, 0.0, float("nan"), float("inf")],
            [-1e6, -1.0, 0.0, 1.0, 1e6],
        ),
        (-1.0, 1e3, -1e3, [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]),
    ],
)
def test__nan_to_norm(
    nan: float,
    posinf: float | None,
    neginf: float | None,
    inputs: list[float],
    expected: list[float],
):
    input_tensor = PhlowerTensor(torch.tensor(inputs))
    phlower_tensors = phlower_tensor_collection({"tensor": input_tensor})

    model = NaNToNum(nan=nan, posinf=posinf, neginf=neginf)

    actual: PhlowerTensor = model(phlower_tensors)

    assert actual.shape == input_tensor.shape
    np.testing.assert_array_almost_equal(
        actual.to_tensor(), torch.tensor(expected)
    )
