import pydantic
import pytest
from hypothesis import given
from hypothesis import strategies as st

from phlower.settings._continue_settings import ContinueSetting


@given(
    stop_count=st.integers(min_value=1, max_value=100),
    lr_factor=st.floats(min_value=0.01, max_value=0.99),
)
def test__parse_lr_decay_continue_setting(stop_count: int, lr_factor: float):
    parameters = {"lr_factor": lr_factor}
    setting_dict = {
        "is_active": True,
        "stop_count": stop_count,
        "continue_type": "lr_decay",
        "parameters": parameters,
    }

    setting = ContinueSetting.model_validate(setting_dict)

    assert setting.is_active is True
    assert setting.stop_count == stop_count
    assert setting.continue_type == "lr_decay"
    assert setting.parameters.lr_factor == lr_factor


@pytest.mark.parametrize(
    "lr_factor, expected_msg",
    [
        (-0.1, "Input should be greater than 0"),
        (0.0, "Input should be greater than 0"),
        (1.0, "Input should be less than 1"),
        (1.5, "Input should be less than 1"),
    ],
)
def test__lr_decay_must_be_positive_and_less_than_1(
    lr_factor: float, expected_msg: str
):

    with pytest.raises(pydantic.ValidationError, match=expected_msg):
        _ = ContinueSetting.model_validate(
            {
                "continue_type": "lr_decay",
                "parameters": {
                    "lr_factor": lr_factor,
                },
            }
        )


def test__parse_optimizer_switch_continue_setting():
    setting_dict = {
        "is_active": True,
        "stop_count": 2,
        "continue_type": "optimizer_switch",
        "parameters": {
            "optimizer_settings": [
                {
                    "optimizer": "Adam",
                    "parameters": {
                        "lr": 0.001,
                        "weight_decay": 0.01,
                    },
                },
                {
                    "optimizer": "SGD",
                    "parameters": {
                        "lr": 0.01,
                        "momentum": 0.9,
                    },
                },
            ]
        },
    }
    setting = ContinueSetting.model_validate(setting_dict)
    assert setting.is_active is True
    assert setting.continue_type == "optimizer_switch"
    assert len(setting.parameters.optimizer_settings) == 2
    assert setting.parameters.optimizer_settings[0].optimizer == "Adam"
    assert setting.parameters.optimizer_settings[0].parameters["lr"] == 0.001
    assert (
        setting.parameters.optimizer_settings[0].parameters["weight_decay"]
        == 0.01
    )
    assert setting.parameters.optimizer_settings[1].optimizer == "SGD"
    assert setting.parameters.optimizer_settings[1].parameters["lr"] == 0.01
    assert (
        setting.parameters.optimizer_settings[1].parameters["momentum"] == 0.9
    )


def test__unmatch_number_of_optimizer_settings():
    setting_dict = {
        "is_active": True,
        "stop_count": 2,
        "continue_type": "optimizer_switch",
        "parameters": {
            "optimizer_settings": [
                {
                    "optimizer": "Adam",
                    "parameters": {
                        "lr": 0.001,
                        "weight_decay": 0.01,
                    },
                }
            ]
        },
    }

    with pytest.raises(
        pydantic.ValidationError,
        match="Length of optimizer_settings must be equal to stop_count",
    ):
        _ = ContinueSetting.model_validate(setting_dict)
