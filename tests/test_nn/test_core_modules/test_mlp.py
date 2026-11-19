from collections.abc import Callable

import numpy as np
import pytest
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import phlower_tensor_collection
from scipy.special import expit, ndtr

from phlower.nn import MLP
from phlower.settings._module_settings import MLPSetting
from tests.test_nn._module_reference import FloatArray, linear


@pytest.mark.parametrize("shape", [(5, 4), (3, 5, 4)])
@pytest.mark.parametrize(
    "activations, reference_activations",
    [
        (["identity", "identity"], (np.positive, np.positive)),
        (["gelu", "identity"], (lambda x: x * ndtr(x), np.positive)),
        (["relu", "tanh"], (lambda x: np.maximum(x, 0), np.tanh)),
        (["tanh", "sigmoid"], (np.tanh, expit)),
    ],
    ids=["identity", "gelu", "relu-tanh", "tanh-sigmoid"],
)
@pytest.mark.parametrize(
    "linear_init",
    [
        None,
        {"init_parameters": {"std": 0.1}},
        {"init_parameters": {"std": 0.05}, "zero_bias": False},
        {"kind": "xavier_uniform", "init_parameters": {"gain": 1.5}},
    ],
    ids=["default-init", "zero-bias", "keep-bias", "xavier-uniform"],
)
def test__configured_mlp_matches_numpy_scipy(
    shape: tuple[int, ...],
    activations: list[str],
    reference_activations: tuple[Callable[[FloatArray], FloatArray], ...],
    linear_init: dict[str, str | bool | dict[str, float]] | None,
) -> None:
    setting = MLPSetting.model_validate(
        {
            "nodes": [4, 8, 6],
            "activations": activations,
            "linear_init": linear_init,
        }
    )
    model = MLP.from_setting(setting)
    values = np.random.default_rng(17).standard_normal(shape, dtype=np.float32)
    inputs = phlower_tensor_collection({"tokens": phlower_tensor(values)})

    result = model(inputs)

    if linear_init is not None and linear_init.get("zero_bias", True):
        for layer in model._chains:
            assert layer.bias is not None
            np.testing.assert_array_equal(layer.bias.detach().numpy(), 0)

    expected = values.astype(np.float64)
    for layer, activation in zip(
        model._chains, reference_activations, strict=True
    ):
        expected = activation(linear(expected, layer))
    np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-6)
