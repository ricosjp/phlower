from __future__ import annotations

import abc
from typing import Annotated, Any, Literal

import pydantic
from pydantic import (
    PlainSerializer,
    PlainValidator,
    SerializationInfo,
    ValidationInfo,
)
from typing_extensions import Self

from phlower.utils.enums import PhlowerIterationSolverType


class IPhlowerIterationSolverSetting(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_target_keys(self) -> list[str]: ...


class EmptySolverSetting(pydantic.BaseModel, IPhlowerIterationSolverSetting):
    def get_target_keys(self) -> list[str]:
        return []

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class SimpleSolverSetting(pydantic.BaseModel, IPhlowerIterationSolverSetting):
    target_keys: list[str] = pydantic.Field(default_factory=list)
    convergence_threshold: float = 100
    max_iterations: int = 100

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.field_validator("convergence_threshold", "max_iterations")
    @classmethod
    def must_be_positive_value(
        cls, value: float | int, info: pydantic.ValidationInfo
    ) -> float | int:
        assert value > 0, f"{info.field_name} must be positive value."
        return value

    @pydantic.field_validator("target_keys")
    @classmethod
    def empty_list_is_not_allowed(cls, value: list[str]) -> list[str]:
        assert (
            len(value) > 0
        ), "Set at least one target item for target_keys in solver settings"

        return value

    def get_target_keys(self) -> list[str]:
        return self.target_keys


class BarzilaiBoweinSolverSetting(
    pydantic.BaseModel, IPhlowerIterationSolverSetting
):
    target_keys: list[str] = pydantic.Field(default_factory=list)
    convergence_threshold: float = 0.01
    max_iterations: int = 100
    divergence_threshold: float = 100
    alpha_component_wise: bool = False
    bb_type: Literal["short", "long"] = "short"

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.field_validator("target_keys")
    @classmethod
    def empty_list_is_not_allowed(cls, value: list[str]) -> list[str]:
        assert (
            len(value) > 0
        ), "Set at least one target item for target_keys in solver settings"

        return value

    @pydantic.field_validator(
        "convergence_threshold", "divergence_threshold", "max_iterations"
    )
    @classmethod
    def must_be_positive_value(
        cls, value: float | int, info: pydantic.ValidationInfo
    ) -> float | int:
        assert value > 0, f"{info.field_name} must be positive value."
        return value

    @pydantic.model_validator(mode="after")
    def convergence_must_be_less_than_divergence(self) -> Self:
        assert (
            self.convergence_threshold < self.divergence_threshold
        ), "convergence threshold must be less than divergence_threshold"
        return self

    def get_target_keys(self) -> list[str]:
        return self.target_keys


_name_to_setting: dict[str, IPhlowerIterationSolverSetting] = {
    PhlowerIterationSolverType.none.value: EmptySolverSetting,
    PhlowerIterationSolverType.simple.value: SimpleSolverSetting,
    PhlowerIterationSolverType.bb.value: BarzilaiBoweinSolverSetting,
}


def _validate(
    vals: dict, info: ValidationInfo
) -> IPhlowerIterationSolverSetting:
    solver_type = info.data["solver_type"]

    if solver_type not in PhlowerIterationSolverType.__members__:
        raise NotImplementedError(
            f"solver_type: {solver_type} is not implemented."
        )

    setting_cls = _name_to_setting[solver_type]
    return setting_cls(**vals)


def _serialize(
    v: pydantic.BaseModel, info: SerializationInfo
) -> dict[str, Any]:
    return v.model_dump()


SolverParameters = Annotated[
    IPhlowerIterationSolverSetting,
    PlainValidator(_validate),
    PlainSerializer(_serialize),
]
