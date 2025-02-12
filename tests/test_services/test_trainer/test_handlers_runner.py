import pathlib

import pytest
from phlower.services.trainer._handlers import HandlersRunner
from phlower.settings import PhlowerSetting
from phlower.utils.typing import AfterEvaluationOutput, PhlowerHandlerType

DATA_DIR = pathlib.Path("tests/test_services/test_trainer/data/handlers")


@pytest.mark.parametrize(
    "file_name, desired", [("no_handlers.yml", 0), ("stop_trigger.yml", 1)]
)
def test__initialize_from_setting_file(file_name: str, desired: int):
    setting = PhlowerSetting.read_yaml(DATA_DIR / file_name)
    runner = HandlersRunner.from_setting(setting.training)

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


@pytest.mark.parametrize(
    "file_name, desired", [("no_handlers.yml", 1), ("stop_trigger.yml", 2)]
)
def test__initialize_from_setting_file_with_user_defined(
    file_name: str, desired: int
):
    setting = PhlowerSetting.read_yaml(DATA_DIR / file_name)
    runner = HandlersRunner.from_setting(
        setting.training, user_defined_handlers={"dummy": DummyCustomHanlder()}
    )

    assert runner.n_handlers == desired


@pytest.mark.parametrize(
    "file_name, threshold",
    [("no_handlers.yml", 1.0), ("stop_trigger.yml", 2.0)],
)
def test__has_termination_flag(file_name: str, threshold: float):
    setting = PhlowerSetting.read_yaml(DATA_DIR / file_name)
    runner = HandlersRunner.from_setting(
        setting.training,
        user_defined_handlers={"dummy": DummyCustomHanlder(threshold)},
    )
    assert not runner.terminate_training

    # Trigger dummy handler
    dummy = AfterEvaluationOutput(train_eval_loss=1.2, validation_eval_loss=3.0)
    runner(dummy)

    assert runner.terminate_training


@pytest.mark.parametrize(
    "file_name, has_user_defined",
    [
        ("no_handlers.yml", False),
        ("stop_trigger.yml", False),
        ("stop_trigger.yml", True),
    ],
)
def test__dump_and_restore(file_name: str, has_user_defined: bool):
    setting = PhlowerSetting.read_yaml(DATA_DIR / file_name)
    if has_user_defined:
        user_defined_handlers = {"dummy": DummyCustomHanlder()}
    else:
        user_defined_handlers = {}

    runner = HandlersRunner.from_setting(
        setting.training, user_defined_handlers=user_defined_handlers
    )

    dumped = runner.state_dict()

    new_runner = HandlersRunner.from_setting(
        setting.training, user_defined_handlers=user_defined_handlers
    )
    new_runner.load_state_dict(dumped)

    assert runner.n_handlers == new_runner.n_handlers
    assert dumped == new_runner.state_dict()
