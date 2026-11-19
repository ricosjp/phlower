import pytest
from pydantic import ValidationError

from phlower.settings._module_settings._linear_init_setting import (
    LinearInitSetting,
    _TruncatedNormalInitParameters,
    _XavierNormalInitParameters,
    _XavierUniformInitParameters,
)


def test__default_linear_initializer() -> None:
    setting = LinearInitSetting()

    assert setting.kind == "truncated_normal"
    assert setting.init_parameters == _TruncatedNormalInitParameters(std=0.02)
    assert setting.init_parameters.to_dict() == {
        "kind": "truncated_normal",
        "std": 0.02,
    }
    assert setting.zero_bias is True


@pytest.mark.parametrize(
    "kind, parameter_type",
    [
        ("xavier_uniform", _XavierUniformInitParameters),
        ("xavier_normal", _XavierNormalInitParameters),
    ],
)
def test__xavier_kind_selects_typed_parameters(
    kind: str,
    parameter_type: type[
        _XavierUniformInitParameters | _XavierNormalInitParameters
    ],
) -> None:
    default = LinearInitSetting.model_validate({"kind": kind})
    assert default.init_parameters == parameter_type(gain=1.0)

    setting = LinearInitSetting.model_validate(
        {
            "kind": kind,
            "init_parameters": {"gain": 1.5},
            "zero_bias": False,
        }
    )
    assert setting.init_parameters == parameter_type(gain=1.5)
    assert setting.init_parameters.to_dict() == {"kind": kind, "gain": 1.5}
    assert not setting.zero_bias
    assert setting.to_dict() == setting.model_dump()
    assert LinearInitSetting.model_validate(setting.model_dump()) == setting


def test__rejects_parameter_model_for_a_different_kind() -> None:
    with pytest.raises(ValidationError, match="init_parameters"):
        LinearInitSetting(
            kind="xavier_normal",
            init_parameters=_XavierUniformInitParameters(),
        )


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"kind": "uniform"}, "truncated_normal"),
        ({"init_parameters": {"std": 0}}, "greater than 0"),
        ({"init_parameters": {"std": -0.01}}, "greater than 0"),
        ({"init_parameters": {"std": float("inf")}}, "finite number"),
        ({"init_parameters": {"std": float("nan")}}, "finite number"),
        ({"init_parameters": {"gain": 1.5}}, "init_parameters.gain"),
        (
            {"kind": "xavier_uniform", "init_parameters": {"std": 0.02}},
            "init_parameters.std",
        ),
        (
            {
                "kind": "xavier_normal",
                "init_parameters": {"gain": float("inf")},
            },
            "finite number",
        ),
        ({"std": 0.02}, "Extra inputs are not permitted"),
    ],
    ids=[
        "unknown-kind",
        "zero-std",
        "negative-std",
        "infinite-std",
        "nan-std",
        "truncated-normal-gain",
        "xavier-std",
        "infinite-gain",
        "flat-parameter",
    ],
)
def test__reject_invalid_linear_initializer(
    parameters: dict[str, str | float | dict[str, float]], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        LinearInitSetting.model_validate(parameters)
