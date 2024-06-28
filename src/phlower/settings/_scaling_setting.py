from __future__ import annotations

import abc
import pathlib
from collections import defaultdict
from typing import Any

import pydantic
from pipe import chain, select
from pydantic import dataclasses as dc

import phlower.utils as utils
from phlower.io import PhlowerDirectory
from phlower.utils.enums import PhlowerScalerName


class IScalerParameter(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def is_parent_scaler(self) -> bool: ...

    @abc.abstractproperty
    def join_fitting(self) -> bool: ...

    @abc.abstractmethod
    def get_scaler_name(self, variable_name: str): ...


class ScalerInputParameters(IScalerParameter, pydantic.BaseModel):
    method: str
    component_wise: bool = True
    parameters: dict[str, Any] = pydantic.Field(default_factory=lambda: {})

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

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

    def get_scaler_name(self, variable_name: str):
        return f"scaler_{variable_name}"


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

    @property
    def join_fitting(self) -> bool:
        return self.join_fitting

    def get_scaler_name(self, variable_name: str):
        return f"scaler_{self.same_as}"


InputParameterSetting = ScalerInputParameters | SameAsInputParameters


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class PhlowerScalingSetting:
    interim_base_directory: pathlib.Path
    varaible_name_to_scalers: dict[
        str, ScalerInputParameters | SameAsInputParameters
    ] = pydantic.Field(default_factory=lambda: {})

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

    def resolve_scalers(self) -> list[ScalerResolvedParameters]:
        return _resolve(self.varaible_name_to_scalers)

    def get_variable_names(self) -> list[str]:
        return list(self.varaible_name_to_scalers.keys())


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class ScalerResolvedParameters(IScalerParameter):
    scaler_name: str
    transform_members: list[str]
    fitting_members: list[str]
    method: str
    component_wise: bool = True
    parameters: dict[str, Any] = pydantic.Field(default_factory=lambda: {})

    @pydantic.field_validator("method")
    @classmethod
    def is_exist_method(cls, v):
        logger = utils.get_logger(__name__)

        names = utils.get_registered_scaler_names()
        if v not in names:
            # NOTE: Just warn in case that users define their own method.
            logger.warning(f"Scaler name: {v} is not implemented.")
        return v

    @pydantic.model_validator(mode="after")
    def _validate_isoam_scaler(self):
        _validate_scaler(self.method, self)
        return self

    def collect_fitting_files(self) -> list[pathlib.Path]:
        directory = PhlowerDirectory()
        file_paths = list(
            self.fitting_members
            | select(lambda x: directory.find_variable_file(x))
            | chain
        )

        return file_paths

    def collect_interim_directories(self) -> list[pathlib.Path]:
        raise NotImplementedError()


# validation functions


def _validate_scaler(
    method_name: str, setting: ScalerResolvedParameters
) -> None:
    if method_name == PhlowerScalerName.ISOAM_SCALE.name:
        _validate_isoam(setting)
        return

    return


def _validate_isoam(setting: ScalerResolvedParameters):
    # validate same_as
    other_components = setting.parameters.get("other_components", [])
    if len(other_components) == 0:
        raise ValueError(
            "To use IsoAMScaler, feed other_components: " f"{other_components}"
        )

    if sorted(other_components) == sorted(setting.fitting_members):
        raise ValueError(
            "In IsoAMScaler, other components does not match fitting variables."
            "Please check join_fitting is correctly set in the varaibles "
            "in other_components"
        )


# endregion

# region


def _resolve(
    variable_name_to_scalers: dict[str, InputParameterSetting],
) -> list[ScalerResolvedParameters]:
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
        ScalerResolvedParameters(
            scaler_name=name,
            transform_members=_scaler_to_transform_items[name],
            fitting_members=_scaler_to_fitting_items[name],
            **setting.__members__,
        )
        for name, setting in _scaler_name_to_setting.items()
    ]

    return _resolved


# endregion
