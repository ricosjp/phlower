from __future__ import annotations

import pathlib
import socket
from collections.abc import Iterable
from functools import cached_property
from typing import Any, Literal, cast

import pydantic
import pydantic.dataclasses as dc
import torch
from pydantic import Field

from phlower.settings._handler_settings import (
    EarlyStoppingSetting,
    HandlerSettingType,
)
from phlower.settings._time_series_sliding_setting import (
    TimeSeriesSlidingSetting,
)
from phlower.utils import (
    OptimizerSelector,
    SchedulerSelector,
    get_logger,
)
from phlower.utils.enums import TrainerInitializeType

_logger = get_logger(__name__)


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

    aggregation_method: Literal["sum", "mean"] = "sum"
    """
    Method to aggregate loss values for all variables.
    Choose from "sum" or "mean". Defaults to "sum".
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

    parameters: dict[str, int | float | bool | str | Any] = Field(
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
                f"{name} is not defined as an optimizer in phlower. "
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

    parameters: dict[str, Any] = Field(default_factory=dict)
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
                f"{name} is not defined as an scheduler in phlower. "
                "If you defined user defined scheduler, "
                "please use `register` function in "
                "`phlower.utils.SchedulerSelector`."
            )
        return name


class TrainerInitializerSetting(pydantic.BaseModel):
    type_name: str | TrainerInitializeType = "none"
    """
    Weight initializer type.
    """

    reference_directory: pathlib.Path | None = None
    """
    Reference directory to get weight.
    It is used when `pretrained` or `restart`
    """

    model_config = pydantic.ConfigDict(
        frozen=True, extra="forbid", validate_default=True
    )

    @pydantic.field_validator("type_name")
    @classmethod
    def check_exist_initializer(cls, name: str) -> str:
        if name not in TrainerInitializeType.__members__:
            raise ValueError(
                f"{name} is not defined initializer for PhlowerTrainer."
                "Choose from none, pretrained, restart."
            )
        return TrainerInitializeType[name]

    @pydantic.field_serializer("type_name")
    @classmethod
    def serialize_type_name(cls, value: str | TrainerInitializeType) -> str:
        if isinstance(value, TrainerInitializeType):
            return value.value
        return value

    @pydantic.field_serializer("reference_directory")
    @classmethod
    def serialize_reference_directory(
        cls, value: pathlib.Path | None
    ) -> str | None:
        if value is None:
            return None
        return str(value)


class ParallelSetting(pydantic.BaseModel):
    is_active: bool = True
    """
    If True, parallel processing is activated.
    """

    parallel_type: Literal["DDP", "FSDP2"] = "DDP"
    """
    Parallel processing type. Defaults to DDP.
    """

    world_size: int = 1
    """
    Number of processes to use. Defaults to 1.
    """

    backend: Literal["nccl", "gloo"] = "nccl"
    """
    Backend to use. Defaults to nccl
    """

    reshard_after_forward: Literal[False] = False
    """
    If True, reshard parameters after forward pass. Defaults to False.
    This controls the parameter behavior after forward and can trade
    off memory and communication.
    This parameter is only used when parallel_type is FSDP2.
    See: https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @cached_property
    def tcp_port(self) -> int:
        port: int = -1
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            addr = s.getsockname()
            port = addr[1]

        assert port != -1, "Failed to get free tcp port"
        return port


class PhlowerTrainerSetting(pydantic.BaseModel):
    loss_setting: LossSetting = Field(
        default_factory=lambda: LossSetting(name2loss={})
    )
    """
    setting for loss function
    """

    optimizer_setting: OptimizerSetting = Field(
        default_factory=lambda: OptimizerSetting()
    )
    """
    setting for optimizer
    """

    scheduler_settings: list[SchedulerSetting] = Field(default_factory=list)
    """
    setting for schedulers
    """

    handler_settings: list[HandlerSettingType] = Field(default_factory=list)
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
    device name. Defaults to cpu. When auto is set, device is automatically
    set to gpu when available, otherwise cpu.
    """

    evaluation_for_training: bool = True
    """
    If True, evaluation for training dataset is performed
    """

    log_every_n_epoch: int = 1
    """
    dump log items every nth epoch
    """

    initializer_setting: TrainerInitializerSetting = Field(
        default_factory=lambda: TrainerInitializerSetting()
    )
    """
    setting for trainer initializer
    """

    lazy_load: bool = True
    """
    If True, data is loaded lazily.
    If False, all data is loaded at once.
    Defaults to True.
    """

    time_series_sliding: TimeSeriesSlidingSetting = Field(
        default_factory=lambda: TimeSeriesSlidingSetting()
    )
    """
    Setting for sliding window for time series data.
    Defaults to inactive.
    """

    parallel_setting: ParallelSetting = Field(
        default_factory=lambda: ParallelSetting(is_active=False)
    )
    """
    Setting for parallel processing
    """

    non_blocking: bool = False
    """
    If True, non_blocking transfer is used when data is transferred to device.
    Defaults to False.
    """

    pin_memory: bool = False
    """
    If True, pin_memory is used in DataLoader.
    Defaults to False.
    """

    evaluate_context_manager: Literal["inference_mode", "no_grad"] = (
        "inference_mode"
    )
    """
    Context manager to use during evaluation.
    Choose from "no_grad" or "inference_mode".
    Defaults to "inference_mode".
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    def get_early_stopping_patience(self) -> int | None:
        for setting in self.handler_settings:
            if setting.handler != "EarlyStopping":
                continue

            setting = cast(EarlyStoppingSetting, setting)
            return setting.get_patience()

        return None

    @pydantic.field_validator("device", mode="before")
    @classmethod
    def _resolve_device(cls, device: str) -> str:
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda:0"

        return "cpu"

    def get_device(self, rank: int | None = None) -> torch.device:
        if self.parallel_setting.is_active:
            if self.parallel_setting.backend == "gloo":
                return torch.device("cpu")
            if rank is None:
                raise ValueError(
                    "rank must be specified in parallel processing"
                )
            return torch.device(f"cuda:{rank}")

        return torch.device(self.device)
