from phlower.services.trainer._continue_updater import (
    OptimizerSwitchContinueUpdator,
)
from phlower.settings import PhlowerTrainerSetting
from phlower.settings._continue_settings import ContinueSetting


def test__get_ouput_directory_name_suffix():
    cont_setting = ContinueSetting(
        continue_type="optimizer_switch",
        stop_count=2,
        parameters={
            "optimizer_settings": [
                {
                    "optimizer": "AdamW",
                    "parameters": {"lr": 1e-4},
                },
                {
                    "optimizer": "SGD",
                    "parameters": {"lr": 1e-3},
                },
            ]
        },
    )

    updator = OptimizerSwitchContinueUpdator(
        continue_setting=cont_setting.parameters
    )
    suffix = updator.get_output_directory_name_suffix(None, 1)
    assert suffix == "_cont1_opt_AdamW"

    suffix = updator.get_output_directory_name_suffix(None, 2)
    assert suffix == "_cont2_opt_SGD"


def test__update_parameters():
    cont_setting = ContinueSetting(
        continue_type="optimizer_switch",
        stop_count=2,
        parameters={
            "optimizer_settings": [
                {
                    "optimizer": "AdamW",
                    "parameters": {"lr": 1e-4},
                },
                {
                    "optimizer": "SGD",
                    "parameters": {"lr": 1e-3},
                },
            ]
        },
    )

    updator = OptimizerSwitchContinueUpdator(
        continue_setting=cont_setting.parameters
    )

    prev_setting = PhlowerTrainerSetting(
        optimizer_setting={
            "optimizer": "Adam",
            "parameters": {"lr": 1e-5},
        },
    )
    new_setting = updator.update_parameters(prev_setting, 1)
    assert new_setting.optimizer_setting.optimizer == "AdamW"
    assert new_setting.optimizer_setting.get_lr() == 1e-4

    new_setting = updator.update_parameters(prev_setting, 2)
    assert new_setting.optimizer_setting.optimizer == "SGD"
    assert new_setting.optimizer_setting.get_lr() == 1e-3
