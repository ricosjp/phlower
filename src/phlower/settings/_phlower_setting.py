from __future__ import annotations

import pathlib
from collections.abc import Iterable

import pydantic
from pydantic import dataclasses as dc
from typing_extensions import Self

from phlower._base import PhysicalDimensionsClass
from phlower.io import PhlowerYamlFile
from phlower.settings._group_settings import GroupModuleSetting
from phlower.settings._scaling_setting import PhlowerScalingSetting
from phlower.utils.enums import ModelSelectionType


class PhlowerSetting(pydantic.BaseModel):
    training: PhlowerTrainerSetting | None = None
    """
    training setting. Defaults to None.
    """

    model: PhlowerModelSetting | None = None
    """
    model setting. Defaults to None.
    """

    scaling: PhlowerScalingSetting | None = None
    """
    scaling setting. Defaults to None.
    """

    prediction: PhlowerPredictorSetting | None = None
    """
    prediction setting. Defaults to None.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    @classmethod
    def read_yaml(
        cls,
        file_path: pathlib.Path | str | PhlowerYamlFile,
        decrypt_key: bytes | None = None,
    ) -> Self:
        """Read yaml file and parse to PhlowerSetting object.

        Args:
            file_path (pathlib.Path | str | PhlowerYamlFile):
                path to yaml file
            decrypt_key (bytes | None, optional):
                key to decrypt file. Defaults to None.

        Returns:
            Self: PhlowerSetting object
        """
        path = PhlowerYamlFile(file_path)
        data = path.load(decrypt_key=decrypt_key)
        return PhlowerSetting(**data)


class PhlowerModelSetting(pydantic.BaseModel):
    variable_dimensions: dict[str, PhysicalDimensionsClass] = pydantic.Field(
        default_factory=lambda: {}, validate_default=True
    )
    """
    dictionary which maps variable name to value
    """

    network: GroupModuleSetting
    """
    define structure of neural network
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )


class PhlowerTrainerSetting(pydantic.BaseModel):
    loss_setting: LossSetting
    """
    setting for loss function
    """

    n_epoch: int = 10
    """
    the number of epochs. Defaults to 10.
    """

    random_seed: int = 0
    """
    random seed. Defaults to 0
    """

    lr: float = 0.001
    """
    learning rate. Defaults to 0.001
    """

    batch_size: int = 1
    """
    batch size. Defaults to 1
    """

    num_workers: int = 1
    """
    the number of cores. Defaults to 1.
    """

    device: str = "cpu"
    """
    device name. Defaults to cpu
    """

    non_blocking: bool = False

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class LossSetting:
    name2loss: dict[str, str]
    """
    Dictionary which maps name of target variable to name of loss function.
    """

    name2weight: dict[str, str] | None = None
    """
    Dictionary which maps weight value to name of output variable.
    Defaults to None. If None, total loss value is calculated as summation of all loss values. 
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


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class PhlowerPredictorSetting:
    selection_mode: str
    """
    Define method to select checkpoint file.
    Choose from "best", "latest", "train_best", "specified"
    """

    device: str = "cpu"
    """
    device name. Defaults to cpu
    """

    log_file_name: str = "log"
    """
    name of log file. Defaults to "log"
    """

    saved_setting_filename: str = "model"
    """
    file name of pretrained model setting. Defaults to "model"
    """

    batch_size: int = 1
    """
    batch size. Defaults to 1
    """

    num_workers: int = 1
    """
    the number of cores. Defaults to 1.
    """

    non_blocking: bool = False

    random_seed: int = 0
    """
    random seed. Defaults to 0
    """

    @pydantic.field_validator("selection_mode")
    @classmethod
    def check_valid_selection_mode(cls, name):
        names = [v.value for v in ModelSelectionType]
        if name not in names:
            raise ValueError(f"{name} selection mode does not exist.")
        return name
