from __future__ import annotations

import pathlib
from collections.abc import Iterable

import pydantic
from pydantic import dataclasses as dc
from typing_extensions import Self

from phlower._base import PhysicalDimensionsClass
from phlower.io import PhlowerYamlFile
from phlower.settings._model_settings import GroupModuleSetting
from phlower.settings._scaling_setting import PhlowerScalingSetting
from phlower.utils.enums import ModelSelectionType


class PhlowerSetting(pydantic.BaseModel):
    training: PhlowerTrainerSetting | None = None
    """
    training setting. Optional(Default: None)
    """

    model: PhlowerModelSetting | None = None
    """
    model setting. Optional(Default: None)
    """
    
    scaling: PhlowerScalingSetting | None = None
    """
    scaling setting. Optional(Default: None)
    """

    prediction: PhlowerPredictorSetting | None = None
    """
    prediction setting. Optional(Default: None)
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

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )


class PhlowerTrainerSetting(pydantic.BaseModel):
    loss_setting: LossSetting
    n_epoch: int = 10
    random_seed: int = 0
    lr: float = 0.001
    batch_size: int = 1
    num_workers: int = 1
    device: str = "cpu"
    non_blocking: bool = False

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class LossSetting:
    name2loss: dict[str, str]
    name2weight: dict[str, str] | None = None

    def loss_names(self) -> Iterable[str]:
        return self.name2loss.values()

    def loss_variable_names(self) -> list[str]:
        return list(self.name2loss.keys())


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class PhlowerPredictorSetting:
    selection_mode: str
    device: str = "cpu"
    log_file_name: str = "log"
    saved_setting_filename: str = "model"

    batch_size: int = 1
    num_workers: int = 1
    device: str = "cpu"
    non_blocking: bool = False
    random_seed: int = 0

    @pydantic.field_validator("selection_mode")
    @classmethod
    def check_valid_selection_mode(cls, name):
        names = [v.value for v in ModelSelectionType]
        if name not in names:
            raise ValueError(f"{name} selection mode does not exist.")
        return name
