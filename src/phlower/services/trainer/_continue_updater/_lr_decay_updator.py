from typing import cast

from phlower.settings._continue_settings import LRDecayContinueSetting
from phlower.settings._trainer_setting import (
    EarlyStoppingSetting,
    PhlowerTrainerSetting,
)

from ._interface import IContinueParameterUpdator


class LRDecayContinueUpdator(IContinueParameterUpdator):
    def __init__(self, continue_setting: LRDecayContinueSetting):
        self._setting = continue_setting

    def get_output_directory_name_suffix(
        self, prev_setting: PhlowerTrainerSetting, continue_count: int
    ) -> str:
        new_lr = _compute_next_lr(
            prev_setting.optimizer_setting.get_lr(),
            self._setting.lr_factor,
        )
        suffix = f"_cont{continue_count}_lr{new_lr:.3e}"
        return suffix

    def update_parameters(
        self, prev_setting: PhlowerTrainerSetting, continue_count: int
    ) -> PhlowerTrainerSetting:
        prev_lr = prev_setting.optimizer_setting.get_lr()
        if prev_lr is None:
            raise ValueError(
                "Learning rate is not set in original setting. "
                "Cannot continue training with lr_decay."
            )

        prev_patience = prev_setting.get_early_stopping_patience()
        if prev_patience is None:
            raise ValueError(
                "Early stopping patience is not set in original setting. "
                "Cannot continue training with lr_decay."
            )

        new_lr = _compute_next_lr(prev_lr, self._setting.lr_factor)
        new_patience = int(prev_patience / self._setting.lr_factor)

        new_optimizer_setting_dict = (
            prev_setting.optimizer_setting.model_dump()
            | {
                "parameters": {
                    **prev_setting.optimizer_setting.parameters,
                    "lr": new_lr,
                }
            }
        )

        prev_early_stopping_setting = [
            v
            for v in prev_setting.handler_settings
            if isinstance(v, EarlyStoppingSetting)
        ]
        assert len(prev_early_stopping_setting) == 1
        prev_early_stopping_setting = cast(
            EarlyStoppingSetting, prev_early_stopping_setting[0]
        )
        new_early_stopping_setting_dict = (
            prev_early_stopping_setting.model_dump()
            | {
                "parameters": {
                    **prev_early_stopping_setting.parameters,
                    "patience": new_patience,
                }
            }
        )

        new_scheduler_settings = [
            new_early_stopping_setting_dict
            if isinstance(v, EarlyStoppingSetting)
            else v
            for v in prev_setting.handler_settings
        ]

        return PhlowerTrainerSetting.model_validate(
            prev_setting.model_dump()
            | {
                "optimizer_setting": new_optimizer_setting_dict,
                "handler_settings": new_scheduler_settings,
            }
        )


def _compute_next_lr(prev_lr: float, lr_factor: float) -> float:
    return prev_lr * lr_factor
