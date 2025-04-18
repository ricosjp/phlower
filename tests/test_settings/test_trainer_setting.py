import pytest
from hypothesis import given
from hypothesis import strategies as st
from phlower.settings import PhlowerTrainerSetting
from phlower.utils import OptimizerSelector, SchedulerSelector


@pytest.mark.parametrize(
    "content, expected",
    [
        ({}, None),
        (
            {
                "handler_settings": [
                    {
                        "handler": "EarlyStopping",
                        "parameters": {"patience": 10},
                    }
                ]
            },
            10,
        ),
        (
            {
                "handler_settings": [
                    {
                        "handler": "EarlyStopping",
                        "parameters": {"patience": 100},
                    }
                ]
            },
            100,
        ),
    ],
)
def test__get_early_stopping_patience(content: dict, expected: int | None):
    setting = PhlowerTrainerSetting(**content)

    if expected is None:
        assert setting.get_early_stopping_patience() is None
    else:
        assert setting.get_early_stopping_patience() == expected


@pytest.mark.parametrize(
    "content", [({"initializer_setting": {"type_name": "user_defined"}})]
)
def test__raise_error_when_unexpected_trainer_initializer_type(
    content: dict,
):
    with pytest.raises(ValueError) as ex:
        _ = PhlowerTrainerSetting(**content)

    assert "is not defined initializer" in str(ex.value)


@pytest.mark.parametrize(
    "content",
    [
        (
            {
                "initializer_setting": {
                    "type_name": "none",
                    "reference_directory": "tmp/somewhere",
                }
            }
        ),
        (
            {
                "initializer_setting": {
                    "type_name": "restart",
                    "reference_directory": "tmp/somewhere",
                }
            }
        ),
        (
            {
                "initializer_setting": {
                    "type_name": "pretrained",
                    "reference_directory": "tmp/somewhere",
                }
            }
        ),
    ],
)
def test__serializer_trainer_initializer(content: dict):
    setting = PhlowerTrainerSetting(**content)

    dumped = setting.model_dump()

    assert isinstance(dumped["initializer_setting"]["type_name"], str)
    assert isinstance(dumped["initializer_setting"]["reference_directory"], str)


@given(st.sampled_from(SchedulerSelector.get_registered_names()))
def test__can_load_defined_schedulers(name: str):
    setting = PhlowerTrainerSetting(scheduler_settings=[{"scheduler": name}])

    assert len(setting.scheduler_settings) == 1
    assert setting.scheduler_settings[0].scheduler == name


@pytest.mark.parametrize("name", ["my_custom_scheduler", "unknown_scheduler"])
def test__raise_error_when_not_implemented_scheduler(name: str):
    with pytest.raises(ValueError) as ex:
        _ = PhlowerTrainerSetting(scheduler_settings=[{"scheduler": name}])
    assert "is not defined as an scheduler in phlower" in str(ex.value)


@given(name=st.sampled_from(OptimizerSelector.get_registered_names()))
def test__can_load_defined_optimizer(name: str):
    setting = PhlowerTrainerSetting(
        optimizer_setting={
            "optimizer": name,
            "parameters": {
                "lr": 0.001,
                "weight_decay": 0.01,
            },
        }
    )

    assert setting.optimizer_setting.optimizer == name
    assert setting.optimizer_setting.parameters["lr"] == 0.001
    assert setting.optimizer_setting.parameters["weight_decay"] == 0.01


@pytest.mark.parametrize(
    "name",
    ["my_custom_optimizer", "unknown_optimizer"],
)
def test__raise_error_when_not_implemented_optimizer(name: str):
    with pytest.raises(ValueError) as ex:
        _ = PhlowerTrainerSetting(
            optimizer_setting={
                "optimizer": name,
                "parameters": {
                    "lr": 0.001,
                    "weight_decay": 0.01,
                },
            }
        )
    assert "is not defined as an optimizer in phlower" in str(ex.value)
