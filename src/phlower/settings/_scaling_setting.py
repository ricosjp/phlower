from __future__ import annotations

import abc
import pathlib
from collections import defaultdict
from typing import Any

import pydantic
import yaml
from pipe import select, where
from pydantic import dataclasses as dc
from typing_extensions import Self

import phlower.utils as utils
from phlower.io import PhlowerDirectory
from phlower.io._files import IPhlowerNumpyFile
from phlower.utils.enums import PhlowerScalerName


class IScalerParameter(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def is_parent_scaler(self) -> bool: ...

    @abc.abstractmethod
    def get_scaler_name(self, variable_name: str): ...


class ScalerInputParameters(IScalerParameter, pydantic.BaseModel):
    method: str
    component_wise: bool = False
    parameters: dict[str, Any] = pydantic.Field(default_factory=lambda: {})
    join_fitting: bool = True

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    @property
    def is_parent_scaler(self) -> bool:
        return True

    @pydantic.field_validator("join_fitting")
    @classmethod
    def must_be_true(cls, v: bool) -> bool:
        if not v:
            raise ValueError("join_fitting must be True except same_as.")
        return v

    @pydantic.field_validator("method")
    @classmethod
    def is_exist_method(cls, v: str) -> str:
        logger = utils.get_logger(__name__)

        names = utils.get_registered_scaler_names()
        if v not in names:
            # NOTE: Just warn in case that users define their own method.
            logger.warning(f"Scaler name: {v} is not implemented.")
        return v

    def get_scaler_name(self, variable_name: str) -> str:
        return f"SCALER_{variable_name}"


class SameAsInputParameters(IScalerParameter, pydantic.BaseModel):
    same_as: str
    join_fitting: bool = False

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

    @property
    def is_parent_scaler(self) -> bool:
        return False

    def get_scaler_name(self, variable_name: str) -> str:
        return f"SCALER_{self.same_as}"


InputParameterSetting = ScalerInputParameters | SameAsInputParameters


class PhlowerScalingSetting(pydantic.BaseModel):
    variable_name_to_scalers: dict[
        str, ScalerInputParameters | SameAsInputParameters | str
    ] = pydantic.Field(default_factory=lambda: {})

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.model_validator(mode="after")
    def validate_scalers(self) -> Self:
        for k, v in self.variable_name_to_scalers.items():
            if isinstance(v, str):
                self.variable_name_to_scalers[k] = ScalerInputParameters(
                    method=v
                )

        for k, v in self.variable_name_to_scalers.items():
            if v.is_parent_scaler:
                continue

            if v.same_as not in self.variable_name_to_scalers:
                raise ValueError(
                    f"'same_as' item; '{v.same_as}' in {k} "
                    "does not exist in preprocess settings."
                )

        return self

    @classmethod
    def read_yaml(cls, file_path: pathlib.Path | str) -> Self:
        with open(file_path) as fr:
            data = yaml.load(fr, yaml.SafeLoader)

        return PhlowerScalingSetting(**data)

    def resolve_scalers(self) -> list[ScalerResolvedParameter]:
        return _resolve(self.variable_name_to_scalers)

    def get_variable_names(self) -> list[str]:
        return list(self.variable_name_to_scalers.keys())

    def is_scaler_exist(self, variable_name: str) -> bool:
        return variable_name in self.variable_name_to_scalers

    def get_scaler_name(self, variable_name: str) -> str | None:
        scaler_setting = self.variable_name_to_scalers.get(variable_name)
        if scaler_setting is None:
            return None

        return scaler_setting.get_scaler_name(variable_name)


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class ScalerResolvedParameter:
    scaler_name: str
    transform_members: list[str]
    fitting_members: list[str]
    method: str
    component_wise: bool = False
    join_fitting: bool = True
    parameters: dict[str, Any] = pydantic.Field(default_factory=lambda: {})

    @pydantic.field_validator("method")
    @classmethod
    def is_exist_method(cls, v: str) -> str:
        logger = utils.get_logger(__name__)

        names = utils.get_registered_scaler_names()
        if v not in names:
            # NOTE: Just warn in case that users define their own method.
            logger.warning(f"Scaler name: {v} is not implemented.")
        return v

    @pydantic.model_validator(mode="after")
    def _validate_isoam_scaler(self) -> Self:
        _validate_scaler(self.method, self)
        return self

    def collect_fitting_files(
        self,
        directory: PhlowerDirectory | pathlib.Path,
        allow_missing: bool = False,
    ) -> list[IPhlowerNumpyFile]:
        _directory = PhlowerDirectory(directory)
        file_paths = list(
            self.fitting_members
            | select(
                lambda x: _directory.find_variable_file(
                    x, allow_missing=allow_missing
                )
            )
        )

        return file_paths

    def collect_transform_files(
        self,
        directory: PhlowerDirectory | pathlib.Path,
        allow_missing: bool = False,
    ) -> list[IPhlowerNumpyFile]:
        _directory = PhlowerDirectory(directory)
        file_paths = list(
            self.transform_members
            | select(
                lambda x: _directory.find_variable_file(
                    x, allow_missing=allow_missing
                )
            )
            | where(lambda x: x is not None)
        )

        return file_paths


# validation functions


def _validate_scaler(
    method_name: str, setting: ScalerResolvedParameter
) -> None:
    if method_name == PhlowerScalerName.ISOAM_SCALE.value:
        _validate_isoam(setting)
        return

    return


def _validate_isoam(setting: ScalerResolvedParameter):
    # validate same_as
    other_components = setting.parameters.get("other_components", [])
    if len(other_components) == 0:
        raise ValueError(
            "To use IsoAMScaler, feed other_components: " f"{other_components}"
        )

    if len(other_components) + 1 != len(setting.fitting_members):
        raise ValueError(
            "In IsoAMScaler, other components does not match fitting variables."
            "Please check join_fitting is correctly set in the variables "
            "in other_components"
        )
    return


# endregion

# region


def _resolve(
    variable_name_to_scalers: dict[str, InputParameterSetting],
) -> list[ScalerResolvedParameter]:
    _scaler_name_to_setting: dict[str, ScalerInputParameters] = {}
    _scaler_to_fitting_items: dict[str, list[str]] = defaultdict(list)
    _scaler_to_transform_items: dict[str, list[str]] = defaultdict(list)

    for name, setting in variable_name_to_scalers.items():
        scaler_name = setting.get_scaler_name(name)
        _scaler_to_transform_items[scaler_name].append(name)
        if setting.join_fitting:
            _scaler_to_fitting_items[scaler_name].append(name)
        if setting.is_parent_scaler:
            _scaler_name_to_setting[scaler_name] = setting

    _resolved = [
        ScalerResolvedParameter(
            scaler_name=name,
            transform_members=_scaler_to_transform_items[name],
            fitting_members=_scaler_to_fitting_items[name],
            **setting.model_dump(),
        )
        for name, setting in _scaler_name_to_setting.items()
    ]

    return _resolved


# endregion
