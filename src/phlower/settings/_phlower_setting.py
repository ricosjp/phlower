from __future__ import annotations

import abc
import pathlib
from collections.abc import Iterable
from typing import Any

import pydantic
import yaml
from pipe import chain, select, where
from pydantic import dataclasses as dc
from typing_extensions import Self

import phlower.utils as utils
from phlower.io import PhlowerDirectory
from phlower.settings._model_settings import GroupModuleSetting


@dc.dataclass(frozen=True)
class PhlowerSetting:
    trainer: PhlowerTrainerSetting
    model: GroupModuleSetting
    scaling: PhlowerScalingSetting

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
    interim_base_directory: pathlib.Path
    varaible_name_to_scalers: dict[str, ScalerParameters | SameAsParameters] = (
        pydantic.Field(default_factory=lambda: {})
    )

    @pydantic.model_validator(mode="after")
    def _check_same_as(self):
        for k, v in self.varaible_name_to_scalers.items():
            if v.is_parent_scaler:
                continue

            if v.same_as not in self.varaible_name_to_scalers:
                raise ValueError(
                    f"'same_as' item; '{v.same_as}' in {k} "
                    "does not exist in preprocess settings."
                )

        return self

    def get_variable_names(self) -> list[str]:
        return list(self.varaible_name_to_scalers.keys())

    def collect_interim_directories(self) -> list[pathlib.Path]:
        raise NotImplementedError()

    def get_effective_scaler_parameters(self) -> dict[str, ScalerParameters]:
        results = {
            self.get_scaler_name(name): v
            for name, v in self.varaible_name_to_scalers.items()
            if not v.is_parent_scaler
        }
        return results

    def get_scaler_name(self, variable_name: str):
        param = self.varaible_name_to_scalers[variable_name]
        if not param.is_parent_scaler:
            return f"scaler_{variable_name}"

        return f"scaler_{param.same_as}"

    def collect_fitting_files(self, scaler_name: str) -> list[pathlib.Path]:
        target_variables: list[str] = list(
            self.varaible_name_to_scalers.items()
            | where(lambda x: self.get_scaler_name(x) == scaler_name)
            | where(lambda k, v: v.join_fitting)
            | select(lambda k, v: k)
        )

        directory = PhlowerDirectory()
        file_paths = list(
            target_variables
            | select(lambda x: directory.find_variable_file(x))
            | chain
        )

        return file_paths


class IScalerParameter(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def is_parent_scaler(self) -> bool: ...

    @abc.abstractproperty
    def join_fitting(self) -> bool: ...


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class ScalerParameters(IScalerParameter):
    method: str
    component_wise: bool = True
    parameters: dict[str, Any] = pydantic.Field(default_factory=lambda: {})

    @property
    def is_parent_scaler(self) -> bool:
        return True

    @property
    def join_fitting(self) -> bool:
        return True

    @pydantic.field_validator("method")
    @classmethod
    def is_exist_method(cls, v):
        logger = utils.get_logger(__name__)

        names = utils.get_registered_scaler_names()
        if v not in names:
            # NOTE: Just warn in case that users define their own method.
            logger.warning(f"Scaler name: {v} is not implemented.")
        return v


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class SameAsParameters(IScalerParameter):
    same_as: str
    join_fitting: bool = False

    @property
    def is_parent_scaler(self) -> bool:
        return False

    @property
    def join_fitting(self) -> bool:
        return self.join_fitting
