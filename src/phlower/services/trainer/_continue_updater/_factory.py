from phlower.settings._continue_settings import (
    ContinueSetting,
)
from phlower.utils.enums import ContinueTrainingType

from ._interface import IContinueParameterUpdator
from ._lr_decay_updator import LRDecayContinueUpdator
from ._optimizer_switch_updator import OptimizerSwitchContinueUpdator


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
            case _:
                raise ValueError(
                    "Invalid continue type value is set. "
                    f"Input: {continue_setting.continue_type}"
                )
