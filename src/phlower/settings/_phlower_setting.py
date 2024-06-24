from __future__ import annotations

import pathlib
from typing import Iterable, Any

import pydantic
import yaml
from pydantic import dataclasses as dc
from typing_extensions import Self

from phlower.settings._model_settings import GroupModuleSetting
from phlower.services.preprocessing import get_registered_scaler_names


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
    lr: float = 0.001

    variable_dimensions: dict[str, dict[str, float]] = pydantic.Field(
        default_factory=lambda: {}
    )


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class LossSetting:
    name2loss: dict[str, str]
    name2weight: dict[str, str] | None = None

    def loss_names(self) -> Iterable[str]:
        return self.name2loss.values()


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class PhlowerScalingSetting:
    name_to_scalers: dict[str, ScalerParameters] = pydantic.Field(
        default_factory=lambda: {}
    )

    @pydantic.model_validator(mode="after")
    def _check_same_as(self):
        for k, v in self.name_to_scalers.items():
            if v.same_as is None:
                continue

            if v.same_as not in self.name_to_scalers:
                raise ValueError(
                    f"'same_as' item; '{v.same_as}' in {k} does not exist in preprocess settings."
                )

        return self

@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class ScalerParameters:
    method: str | None = None
    component_wise: bool = True
    parameters: dict[str, Any] = pydantic.Field(
        default_factory=lambda: {}
    )
    same_as: str | None = None

    @pydantic.model_validator()
    @classmethod
    def _not_allowed_both_None(self):
        is_none_method = self.method is None
        is_none_same = self.same_as is None

        if is_none_method and is_none_same:
            raise ValueError(
                "method or same_as must be set."
            )
        return self


    @pydantic.field_validator("method")
    @classmethod
    def is_exist_method(cls, v):
        names = get_registered_scaler_names()
        if v not in names:
            raise ValueError(f"Scaler name: {v} is not implemented.")
        return v
