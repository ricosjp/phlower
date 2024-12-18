import hypothesis.strategies as st
import numpy as np
import pytest
import torch
from hypothesis import given
from phlower import PhlowerTensor, phlower_tensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import SPMM
from phlower.settings._module_settings import SPMMSetting


def test__can_call_parameters():
    model = SPMM(support_name="support", factor=1.0)

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize("support_name", ["aaa", "support1"])
@given(
    factor=st.floats(width=32, allow_nan=False, allow_infinity=False),
    tranpose=st.booleans(),
)
def test__can_pass_parameters_via_setting(
    support_name: str, factor: float, tranpose: bool
):
    setting = SPMMSetting(
        support_name=support_name, factor=factor, transpose=tranpose
    )
    model = SPMM.from_setting(setting)

    assert model._factor == factor
    assert model._support_name == support_name
    assert model._transpose is tranpose


@pytest.mark.parametrize(
    "size, is_time_series, factor",
    [
        ((10, 1), False, 1.0),
        ((10, 16), False, 2.0),
        ((10, 3, 16), False, 0.5),
        ((4, 10, 1), True, 3.0),
        ((4, 10, 16), True, 0.0),
        ((4, 10, 3, 16), True, 1.1),
    ],
)
def test__tensor_shape_for_spmm(
    size: tuple[int], is_time_series: bool, factor: float
):
    input_tensor = phlower_tensor(
        torch.rand(*size), is_time_series=is_time_series
    )
    _n = input_tensor.n_vertices()
    support_tensor = phlower_tensor(torch.rand(_n, _n).to_sparse())

    inputs = phlower_tensor_collection({"tensor": input_tensor})
    dict_supports = {"support": support_tensor}
    model = SPMM(support_name="support", factor=factor)

    actual: PhlowerTensor = model(inputs, field_data=dict_supports)

    assert actual.shape == size
    assert actual.is_time_series == is_time_series


@pytest.mark.parametrize(
    "size, is_time_series, factor",
    [
        ((10, 1), False, 1.0),
        ((10, 16), False, 2.0),
        ((10, 3, 16), False, 0.5),
        ((4, 10, 1), True, 3.0),
        ((4, 10, 16), True, 0.0),
        ((4, 10, 3, 16), True, 1.1),
    ],
)
def test__spmm_when_identity_support(
    size: tuple[int], is_time_series: bool, factor: float
):
    input_tensor = phlower_tensor(
        torch.rand(*size), is_time_series=is_time_series
    )
    _n = input_tensor.n_vertices()
    support_tensor = phlower_tensor(torch.eye(_n, _n).to_sparse())

    inputs = phlower_tensor_collection({"tensor": input_tensor})
    dict_supports = {"support": support_tensor}
    model = SPMM(support_name="support", factor=factor)

    actual: PhlowerTensor = model(inputs, field_data=dict_supports)

    assert actual.shape == size
    assert actual.is_time_series == is_time_series

    desired = factor * input_tensor

    np.testing.assert_array_almost_equal(
        actual.to_tensor(), desired.to_tensor()
    )


@pytest.mark.parametrize(
    "size, factor",
    [
        ((10, 1), 1.0),
        ((10, 16), 2.0),
    ],
)
def test__spmm_vertexwise_tensor_when_transpose(
    size: tuple[int], factor: float
):
    input_tensor = phlower_tensor(torch.rand(*size), is_time_series=False)
    _n = input_tensor.n_vertices()
    support_tensor = phlower_tensor(torch.rand(_n, _n).to_sparse())

    inputs = phlower_tensor_collection({"tensor": input_tensor})
    dict_supports = {"support": support_tensor}
    model = SPMM(support_name="support", factor=factor, transpose=True)
    assert model._transpose

    actual = model.forward(inputs, field_data=dict_supports)
    assert actual.shape == size

    desired = factor * torch.sparse.mm(
        torch.transpose(support_tensor, 0, 1), input_tensor
    )

    np.testing.assert_array_almost_equal(
        actual.to_tensor(), desired.to_tensor()
    )
