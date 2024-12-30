import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from phlower import phlower_tensor
from phlower.nn._core_modules._activations import ActivationSelector
from phlower.utils.enums import ActivationType


@given(activation_type=st.sampled_from(ActivationType))
def test__registered_activation_name(activation_type: ActivationType):
    _ = ActivationSelector.select(activation_type.name)


@pytest.mark.parametrize("name", ["sample", "dummy"])
def test__raise_error_when_undefined_activation(name: str):
    with pytest.raises(NotImplementedError) as ex:
        _ = ActivationSelector.select(name)

    assert f"{name} is not implemented" in str(ex.value)


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
