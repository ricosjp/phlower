from phlower.settings._continue_settings import (
    OptimizerSwitchContinueSetting,
)
from phlower.settings._trainer_setting import (
    PhlowerTrainerSetting,
)

from ._interface import IContinueParameterUpdator


class OptimizerSwitchContinueUpdator(IContinueParameterUpdator):
    def __init__(self, continue_setting: OptimizerSwitchContinueSetting):
        self._setting = continue_setting

    def get_output_directory_name_suffix(
        self, prev_setting: PhlowerTrainerSetting, continue_count: int
    ) -> str:
        optim_name = self._setting.optimizer_settings[
            continue_count - 1
        ].optimizer
        suffix = f"_cont{continue_count}_opt_{optim_name}"
        return suffix

    def update_parameters(
        self, prev_setting: PhlowerTrainerSetting, continue_count: int
    ) -> PhlowerTrainerSetting:
        new_optimizer_settings = self._setting.optimizer_settings[
            continue_count - 1
        ]

        return PhlowerTrainerSetting.model_validate(
            prev_setting.model_dump()
            | {
                "optimizer_setting": new_optimizer_settings,
            }
        )
