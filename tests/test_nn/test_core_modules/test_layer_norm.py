import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import LayerNorm
from phlower.settings._module_settings import LayerNormSetting


def test__can_call_parameters():
    model = LayerNorm(nodes=[10, 10])

    # To check LayerNorm inherit torch.nn.Module appropriately
    _ = model.parameters()


@given(
    nodes=st.integers(min_value=1, max_value=10).map(lambda x: [x, x]),
    eps=st.floats(min_value=1e-10, max_value=1e-3),
    elementwise_affine=st.booleans(),
    bias=st.booleans(),
)
def test__can_pass_parameters_via_setting(
    nodes: list[int],
    eps: float,
    elementwise_affine: bool,
    bias: bool,
):
    setting = LayerNormSetting(
        nodes=nodes,
        eps=eps,
        elementwise_affine=elementwise_affine,
        bias=bias,
    )
    model = LayerNorm.from_setting(setting)

    assert model._nodes == nodes
    assert model._layer_norm.eps == eps
    assert model._layer_norm.elementwise_affine == elementwise_affine
    assert (model._layer_norm.bias is not None) == (bias and elementwise_affine)


@pytest.mark.parametrize(
    "input_shape",
    [
        (10, 1),
        (10, 16),
        (10, 3, 16),
        (4, 10, 1),
        (4, 10, 16),
        (4, 10, 3, 16),
    ],
)
def test__layer_norm(input_shape: tuple[int]):
    phlower_tensor = PhlowerTensor(torch.rand(*input_shape))
    phlower_tensors = phlower_tensor_collection({"tensor": phlower_tensor})

    model = LayerNorm(nodes=[input_shape[-1], input_shape[-1]])

    actual: PhlowerTensor = model(phlower_tensors)

    assert actual.shape == input_shape
