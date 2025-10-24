import hypothesis.strategies as st
import numpy as np
import phlower
import pytest
import torch
from hypothesis import given
from phlower import PhlowerTensor, phlower_tensor
from phlower.nn._functionals._activations import ActivationSelector
from phlower.utils.enums import ActivationType
from phlower.utils.exceptions import DimensionIncompatibleError


@given(activation_type=st.sampled_from(ActivationType))
def test__registered_activation_name(activation_type: ActivationType):
    _ = ActivationSelector.select(activation_type.name)


@pytest.mark.parametrize("name", ["sample", "dummy"])
def test__raise_error_when_undefined_activation(name: str):
    with pytest.raises(NotImplementedError) as ex:
        _ = ActivationSelector.select(name)

    assert f"{name} is not implemented" in str(ex.value)


@pytest.mark.parametrize("name", ["sigmoid", "tanh", "smooth_leaky_relu"])
def test__raise_error_when_nonlinear_activation_with_dimensioned_tensor(
    name: str,
):
    tensor = phlower_tensor(torch.randn(10), dimension={"L": 1, "T": -1})
    activation = ActivationSelector.select(name)
    with pytest.raises(DimensionIncompatibleError):
        activation(tensor)


@pytest.mark.parametrize("name", ["sample", "dummy"])
def test__raise_error_when_undefined_inverse_activation(name: str):
    with pytest.raises(NotImplementedError) as ex:
        _ = ActivationSelector.select_inverse(name)

    assert f"{name} is not implemented" in str(ex.value)


@pytest.mark.parametrize(
    "name", ["identity", "leaky_relu0p5", "smooth_leaky_relu", "tanh"]
)
@pytest.mark.parametrize("input_shape", [(3, 4), (5, 8), (1, 10, 3, 1)])
def test__forward_and_inverse(name: str, input_shape: tuple[int]):
    activation = ActivationSelector.select(name)
    inverse_activation = ActivationSelector.select_inverse(name)

    tensor = phlower_tensor(np.random.rand(*input_shape))
    result = inverse_activation(activation(tensor))

    np.testing.assert_array_almost_equal(tensor.to_numpy(), result.to_numpy())


def test_leaky_relu0p5_inverse_leaky_relu0p5():
    x = phlower_tensor(np.random.rand(100)) * 4 - 2.0
    y = phlower.nn.functional.inversed_leaky_relu0p5(
        phlower.nn.functional.leaky_relu0p5(x)
    )
    np.testing.assert_almost_equal(x.to_numpy(), y.to_numpy())


def test_tanh_truncated_atanh():
    x = phlower_tensor(np.random.rand(100)) * 4 - 2
    y = phlower.nn.functional.truncated_atanh(torch.tanh(x))
    np.testing.assert_almost_equal(x.to_numpy(), y.to_numpy(), decimal=6)


def test_smooth_leaky_relu_inverse():
    x = phlower_tensor(np.random.rand(100)) * 4 - 2
    f = phlower.nn.functional.SmoothLeakyReLU()
    y = f.inverse(f(x))
    np.testing.assert_almost_equal(x.to_numpy(), y.to_numpy(), decimal=5)


def test__gelu_with_phlower_tensor():
    x = phlower_tensor(np.random.rand(100))
    y = phlower.nn.functional.gelu(x)
    assert isinstance(y, PhlowerTensor)
    assert y.shape == x.shape
