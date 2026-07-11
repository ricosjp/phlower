from phlower.settings._continue_settings import (
    ContinueSetting,
)
from phlower.utils.enums import ContinueTrainingType

from ._interface import IContinueParameterUpdator
from ._lr_decay_updator import LRDecayContinueUpdator
from ._optimizer_switch_updator import OptimizerSwitchContinueUpdator


class NoneContinueUpdator(IContinueParameterUpdator):
    def update_parameters(
        self,
        prev_setting: ContinueSetting,
        continue_count: int,
    ) -> ContinueSetting:
        raise ValueError(
            "Continue training is not allowed. "
            "Please set continue_type to 'lr_decay' or 'optimizer_switch'."
        )

    def get_output_directory_name_suffix(
        self, prev_setting: ContinueSetting, continue_count: int
    ) -> str:
        raise ValueError(
            "Continue training is not allowed. "
            "Please set continue_type to 'lr_decay' or 'optimizer_switch'."
        )


class ContinueParameterUpdatorFactory:
    @staticmethod
    def create(
        continue_setting: ContinueSetting,
    ) -> IContinueParameterUpdator:
        match continue_setting.continue_type:
            case ContinueTrainingType.lr_decay:
                return LRDecayContinueUpdator(continue_setting.parameters)
            case ContinueTrainingType.optimizer_switch:
                return OptimizerSwitchContinueUpdator(
                    continue_setting.parameters
                )
            case ContinueTrainingType.none:
                return NoneContinueUpdator()
            case _:
                raise ValueError(
                    "Invalid continue type value is set. "
                    f"Input: {continue_setting.continue_type}"
                )
