from unittest.mock import patch

import pytest
import torch

from phlower.nn import MLPConfiguration
from phlower.nn._core_modules._utils import ExtendedLinearList
from phlower.settings._module_settings._linear_init_setting import (
    LinearInitSetting,
)


@pytest.mark.parametrize(
    "bias, zero_bias", [(True, True), (True, False), (False, True)]
)
@pytest.mark.parametrize(
    "parameters, initializer, kwargs",
    [
        ({"init_parameters": {"std": 0.03}}, "trunc_normal_", {"std": 0.03}),
        (
            {"kind": "xavier_uniform", "init_parameters": {"gain": 1.5}},
            "xavier_uniform_",
            {"gain": 1.5},
        ),
        ({"kind": "xavier_normal"}, "xavier_normal_", {"gain": 1.0}),
    ],
    ids=["truncated-normal", "xavier-uniform", "xavier-normal"],
)
def test__from_config_initializes_linear_layers(
    bias: bool,
    zero_bias: bool,
    parameters: dict[str, str | dict[str, float]],
    initializer: str,
    kwargs: dict[str, float],
) -> None:
    config = MLPConfiguration(
        nodes=[4, 8, 6],
        activations=["gelu", "identity"],
        dropouts=[0, 0],
        bias=bias,
        linear_init=LinearInitSetting.model_validate(
            {**parameters, "zero_bias": zero_bias}
        ),
    )

    with (
        patch.object(
            torch.nn.init,
            initializer,
            wraps=getattr(torch.nn.init, initializer),
        ) as init_weights,
        patch.object(
            torch.nn.init, "zeros_", wraps=torch.nn.init.zeros_
        ) as init_biases,
    ):
        model = ExtendedLinearList.from_config(config)

    assert init_weights.call_count == len(model)
    actual_weights = {
        id(call.args[0]): call.kwargs for call in init_weights.call_args_list
    }
    assert actual_weights == {id(layer.weight): kwargs for layer in model}
    actual_biases = sorted(
        id(call.args[0]) for call in init_biases.call_args_list
    )
    expected_biases = sorted(
        id(layer.bias)
        for layer in model
        if zero_bias and layer.bias is not None
    )
    assert actual_biases == expected_biases


@pytest.mark.parametrize(
    "nodes, activations, dropouts",
    [
        ([10, 20, 30], ["tanh", "tanh"], [0.2, 0.1]),
        ([10, 30], ["identity"], [0.3]),
        ([5, 10, 20, 5], ["relu", "relu", "tanh"], [0.3, 0.2, 0.1]),
        ([-1, 20, 30], ["tanh", "tanh"], [0.2, 0.1]),
    ],
)
@pytest.mark.parametrize(
    "linear_init, serialized_init",
    [
        (None, None),
        (
            LinearInitSetting.model_validate(
                {"init_parameters": {"std": 0.03}, "zero_bias": False}
            ),
            {
                "kind": "truncated_normal",
                "init_parameters": {"std": 0.03},
                "zero_bias": False,
            },
        ),
        (
            LinearInitSetting.model_validate(
                {"kind": "xavier_uniform", "init_parameters": {"gain": 1.5}}
            ),
            {
                "kind": "xavier_uniform",
                "init_parameters": {"gain": 1.5},
                "zero_bias": True,
            },
        ),
    ],
    ids=["default-init", "truncated-normal", "xavier-uniform"],
)
def test__to_dict_serializes_linear_init(
    nodes: list[int],
    activations: list[str],
    dropouts: list[float],
    linear_init: LinearInitSetting | None,
    serialized_init: dict[str, str | bool | dict[str, float]] | None,
) -> None:
    config = MLPConfiguration(
        nodes=nodes,
        activations=activations,
        dropouts=dropouts,
        linear_init=linear_init,
    )

    assert config.to_dict() == {
        "nodes": nodes,
        "activations": activations,
        "dropouts": dropouts,
        "bias": True,
        "linear_init": serialized_init,
    }
