from __future__ import annotations

import abc
from typing import Annotated, Any

import pydantic
from pydantic import (
    PlainSerializer,
    PlainValidator,
    SerializationInfo,
    ValidationInfo,
)

from phlower.utils.enums import PhlowerIterationSolverType


class IPhlowerNonlinearSolver(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_target_keys(self) -> list[str]: ...


class SimpleSolverSetting(pydantic.BaseModel, IPhlowerNonlinearSolver):
    target_keys: list[str] = []
    max_iterations: int

    def get_target_keys(self) -> list[str]:
        return self.target_keys


class BarzilaiBoweinSolverSetting(pydantic.BaseModel, IPhlowerNonlinearSolver):
    target_keys: list[str] = []
    max_iterations: int
    divergence_threshold: float
    learn_alpha: bool
    alpha_component_wise: bool
    bb_type: str

    def get_target_keys(self) -> list[str]:
        return self.target_keys


_name_to_setting: dict[str, IPhlowerNonlinearSolver] = {
    PhlowerIterationSolverType.barzilai_borwein.name: (
        BarzilaiBoweinSolverSetting
    )
}


def _validate(vals: dict, info: ValidationInfo) -> IPhlowerNonlinearSolver:
    name = info.data["nn_type"]
    setting_cls = _name_to_setting[name]
    return setting_cls(**vals)


def _serialize(
    v: pydantic.BaseModel, info: SerializationInfo
) -> dict[str, Any]:
    return v.model_dump()


PhlowerModuleParameters = Annotated[
    IPhlowerNonlinearSolver,
    PlainValidator(_validate),
    PlainSerializer(_serialize),
]
