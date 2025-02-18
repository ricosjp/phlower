from __future__ import annotations

from collections.abc import Iterable

import pydantic
import pydantic.dataclasses as dc
from pydantic import Field

from phlower.settings._handler_settings import (
    HandlerSettingType,
)
from phlower.utils import OptimizerSelector, SchedulerSelector


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class LossSetting:
    name2loss: dict[str, str]
    """
    Dictionary which maps name of target variable to name of loss function.
    """

    name2weight: dict[str, float] | None = None
    """
    Dictionary which maps weight value to name of output variable.
    Defaults to None. If None, total loss value is calculated
     as summation of all loss values.
    """

    def loss_names(self) -> Iterable[str]:
        """get registered loss names

        Returns:
            Iterable[str]: names of loss functions
        """
        return self.name2loss.values()

    def loss_variable_names(self) -> list[str]:
        """get variable names

        Returns:
            list[str]: variable names
        """
        return list(self.name2loss.keys())


class OptimizerSetting(pydantic.BaseModel):
    optimizer: str = "Adam"
    """
    Optimizer Class name defined in torch.optim. Default to Adam.
    Ex. Adam, RMSprop, SGD
    """

    parameters: dict[str, int | float | bool | str] = Field(
        default_factory=dict
    )
    """
    Parameters to pass when optimizer class is initialized.
    Allowed parameters depend on the optimizer you choose.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    def get_lr(self) -> float | None:
        return self.parameters.get("lr")

    @pydantic.field_validator("optimizer")
    @classmethod
    def check_exist_scheduler(cls, name: str) -> str:
        if not OptimizerSelector.exist(name):
            raise ValueError(
                f"{name} is not defined in phlower. "
                "If you defined user defined optimizer, "
                "please use `register` function in "
                "`phlower.utils.OptimizerSelector`."
            )
        return name


class SchedulerSetting(pydantic.BaseModel):
    scheduler: str
    """
    Scheduler Class name defined in torch.optim.lr_schedulers.
    """

    parameters: dict[str, int | float | bool | str] = Field(
        default_factory=dict
    )
    """
    Parameters to pass when scheduler class is initialized.
    Allowed parameters depend on the scheduler you choose.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("scheduler")
    @classmethod
    def check_exist_scheduler(cls, name: str) -> str:
        if not SchedulerSelector.exist(name):
            raise ValueError(
                f"{name} is not defined in phlower. "
                "If you defined user defined scheduler, "
                "please use `register` function in "
                "`phlower.utils.SchedulerSelector`."
            )
        return name


class PhlowerTrainerSetting(pydantic.BaseModel):
    loss_setting: LossSetting
    """
    setting for loss function
    """

    optimizer_setting: OptimizerSetting = Field(
        default_factory=OptimizerSetting
    )
    """
    setting for optimizer
    """

    scheduler_setting: list[SchedulerSetting] = Field(default_factory=list)
    """
    setting for schedulers
    """

    handler_setting: list[HandlerSettingType] = Field(default_factory=list)
    """
    setting for handlers
    """

    n_epoch: int = 10
    """
    the number of epochs. Defaults to 10.
    """

    random_seed: int = 0
    """
    random seed. Defaults to 0
    """

    batch_size: int = 1
    """
    batch size. Defaults to 1
    """

    num_workers: int = 0
    """
    the number of cores. Defaults to 0.
    """

    device: str = "cpu"
    """
    device name. Defaults to cpu
    """

    evaluation_for_training: bool = True
    """
    If True, evaluation for training dataset is performed
    """

    log_every_n_epoch: int = 1
    """
    dump log items every nth epoch
    """

    non_blocking: bool = False

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
