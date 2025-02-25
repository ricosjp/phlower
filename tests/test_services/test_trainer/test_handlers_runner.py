import pathlib

import pytest
from phlower.services.trainer._handlers import (
    PhlowerHandlersFactory,
    PhlowerHandlersRunner,
)
from phlower.settings import PhlowerSetting
from phlower.utils.typing import AfterEvaluationOutput, PhlowerHandlerType

DATA_DIR = pathlib.Path("tests/test_services/test_trainer/data/handlers")


@pytest.mark.parametrize("name", ["aaaaa", "not_exsited"])
def test__create_handler_with_not_defined_name(name: str):
    with pytest.raises(NotImplementedError) as ex:
        _ = PhlowerHandlersFactory.create(name, {})
    assert f"Handler {name} is not registered." in str(ex.value)


@pytest.mark.parametrize(
    "file_name, desired", [("no_handlers.yml", 0), ("stop_trigger.yml", 1)]
)
def test__initialize_from_setting_file(file_name: str, desired: int):
    setting = PhlowerSetting.read_yaml(DATA_DIR / file_name)
    runner = PhlowerHandlersRunner.from_setting(setting.training)

    assert runner.n_handlers == desired


class DummyCustomHanlder(PhlowerHandlerType):
    def __init__(self, threshold: float = 1.0):
        self._threshold = threshold

    def __call__(self, output: AfterEvaluationOutput) -> dict:
        if output.train_eval_loss > 1.0:
            return {"TERMINATE": True}

        return {}

    def state_dict(self) -> dict:
        return {"threshold": self._threshold}

    def load_state_dict(self, state_dict: dict):
        self._threshold = state_dict["threshold"]


@pytest.fixture
def setup_user_handler():
    PhlowerHandlersFactory.register("dummy", DummyCustomHanlder)
    yield None
    PhlowerHandlersFactory.unregister("dummy")


@pytest.mark.parametrize(
    "file_name, desired",
    [("user_handlers_1.yml", 1), ("user_handlers_2.yml", 2)],
)
def test__initialize_from_setting_file_with_user_defined(
    file_name: str, desired: int, setup_user_handler: None
):
    setting = PhlowerSetting.read_yaml(DATA_DIR / file_name)
    runner = PhlowerHandlersRunner.from_setting(setting.training)

    assert runner.n_handlers == desired


@pytest.mark.parametrize(
    "file_name",
    [("user_handlers_1.yml"), ("user_handlers_2.yml")],
)
def test__has_termination_flag(file_name: str, setup_user_handler: None):
    setting = PhlowerSetting.read_yaml(DATA_DIR / file_name)
    runner = PhlowerHandlersRunner.from_setting(setting.training)
    assert not runner.terminate_training

    # Trigger dummy handler
    dummy = AfterEvaluationOutput(
        epoch=30,
        train_eval_loss=1.2,
        validation_eval_loss=3.0,
        elapsed_time=100,
    )
    runner(dummy)

    assert runner.terminate_training


@pytest.mark.parametrize(
    "file_name",
    [
        ("no_handlers.yml"),
        ("stop_trigger.yml"),
        ("user_handlers_1.yml"),
        ("user_handlers_2.yml"),
    ],
)
def test__dump_and_restore(file_name: str, setup_user_handler: None):
    setting = PhlowerSetting.read_yaml(DATA_DIR / file_name)
    runner = PhlowerHandlersRunner.from_setting(setting.training)

    dumped = runner.state_dict()

    new_runner = PhlowerHandlersRunner.from_setting(setting.training)
    new_runner.load_state_dict(dumped)

    assert runner.n_handlers == new_runner.n_handlers
    assert dumped == new_runner.state_dict()
