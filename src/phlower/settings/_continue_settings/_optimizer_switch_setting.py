import pydantic

from phlower.settings._optimizer_setting import (
    OptimizerSetting,
)

from ._interface import IContinueParameter, IReadOnlyContinueSetting


class OptimizerSwitchContinueSetting(pydantic.BaseModel, IContinueParameter):
    optimizer_settings: list[OptimizerSetting] = pydantic.Field(
        default_factory=list
    )
    """
    List of optimizer settings to update.
    """

    def validate(self, parent: IReadOnlyContinueSetting) -> None:
        if len(self.optimizer_settings) != parent.stop_count:
            raise ValueError(
                f"Length of optimizer_settings must be equal to stop_count. "
                f"stop_count: {parent.stop_count}, "
                f"length of optimizer_settings: {len(self.optimizer_settings)}"
            )

        return

    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )
