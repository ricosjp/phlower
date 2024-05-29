from __future__ import annotations

import yaml
import pathlib
from typing import Iterable
from typing_extensions import Self

import pydantic
from pydantic import dataclasses as dc

from phlower.settings._model_settings import GroupModuleSetting


@dc.dataclass(frozen=True)
class PhlowerSetting:
    trainer: PhlowerTrainerSetting
    model: GroupModuleSetting

    @classmethod
    def read_yaml(cls, file_path: pathlib.Path | str) -> Self:
        with open(file_path) as fr:
            data = yaml.load(fr, yaml.SafeLoader)
        
        return PhlowerSetting(**data)

@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class PhlowerTrainerSetting:
    loss_setting: LossSetting
    random_seed: int = 0
    batch_size: int = 1
    num_workers: int = 1
    device: str = "cpu"
    non_blocking: bool = False

    variable_dimensions: dict[str, dict[str, float]] = pydantic.Field(
        default_factory=lambda: {}
    )


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class LossSetting:
    name2loss: dict[str, str]
    name2weight: dict[str, str] | None = None

    def loss_names(self) -> Iterable[str]:
        return self.name2loss.values()
