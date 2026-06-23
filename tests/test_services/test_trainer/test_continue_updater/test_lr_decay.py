from phlower.services.trainer._continue_updater import (
    LRDecayContinueUpdator,
)
from phlower.settings import PhlowerTrainerSetting
from phlower.settings._continue_settings import ContinueSetting


def test__get_ouput_directory_name_suffix():
    cont_setting = ContinueSetting(
        continue_type="lr_decay",
        parameters={"lr_factor": 0.5},
    )

    updator = LRDecayContinueUpdator(continue_setting=cont_setting.parameters)
    setting = PhlowerTrainerSetting(
        optimizer_setting={"optimizer": "Adam", "parameters": {"lr": 1e-4}},
        handler_settings=[
            {"handler": "EarlyStopping", "parameters": {"patience": 10}}
        ],
    )

    # cont 1
    suffix = updator.get_output_directory_name_suffix(setting, 1)
    assert suffix == "_cont1_lr5.000e-05"
    setting = updator.update_parameters(setting, 1)
    assert setting.optimizer_setting.parameters["lr"] == 5e-5
    early_stopping_settings = [
        v for v in setting.handler_settings if v.handler == "EarlyStopping"
    ]
    assert len(early_stopping_settings) == 1
    assert early_stopping_settings[0].parameters["patience"] == 20

    # cont 2
    suffix = updator.get_output_directory_name_suffix(setting, 2)
    assert suffix == "_cont2_lr2.500e-05"
    setting = updator.update_parameters(setting, 2)
    assert setting.optimizer_setting.parameters["lr"] == 2.5e-5
    early_stopping_settings = [
        v for v in setting.handler_settings if v.handler == "EarlyStopping"
    ]
    assert len(early_stopping_settings) == 1
    assert early_stopping_settings[0].parameters["patience"] == 40
