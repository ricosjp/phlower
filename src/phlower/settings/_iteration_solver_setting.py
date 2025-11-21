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
from typing_extensions import Self, deprecated

from phlower.utils.enums import PhlowerIterationSolverType


class IPhlowerIterationSolverSetting(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_update_keys(self) -> list[str]: ...


class EmptySolverSetting(pydantic.BaseModel, IPhlowerIterationSolverSetting):
    def get_update_keys(self) -> list[str]:
        return []

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class SimpleSolverSetting(pydantic.BaseModel, IPhlowerIterationSolverSetting):
    target_keys: list[str] = pydantic.Field(
        default_factory=list,
        deprecated=deprecated(
            "target_keys is deprecated. "
            "Use update_keys and residual_keys instead."
        ),
        exclude=True,
    )
    """
    Deprecated: List of variable names to be optimized.
    Use update_keys and residual_keys instead.
    """

    convergence_threshold: float = pydantic.Field(100, gt=0)
    """
    Convergence threshold for the solver.
    """

    max_iterations: int = pydantic.Field(100, gt=0)
    """
    Maximum number of iterations for the solver.
    """

    update_keys: list[str] = pydantic.Field(default_factory=list)
    """
    List of variable names to be updated by the solver.
    If empty list is given, update_keys are set to be same as target_keys.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.model_validator(mode="before")
    def pass_target_keys_to_update_and_residual_keys(
        cls, vals: dict[str, Any]
    ) -> dict[str, Any]:
        if not vals.get("target_keys"):
            # None or empty target_keys
            return vals

        target_keys = vals["target_keys"]
        if vals.get("update_keys"):
            raise ValueError(
                "Cannot set both target_keys and update_keys. "
                "target_keys is deprecated. "
                "Use update_keys instead."
            )

        vals["update_keys"] = target_keys
        vals.pop("target_keys")

        return vals

    @pydantic.model_validator(mode="after")
    def empty_list_is_not_allowed(self) -> Self:
        assert (
            len(self.update_keys) > 0
        ), "Set at least one target item for target_keys in solver settings"

        return self

    def get_update_keys(self) -> list[str]:
        return self.update_keys


class BarzilaiBoweinSolverSetting(
    pydantic.BaseModel, IPhlowerIterationSolverSetting
):
    target_keys: list[str] = pydantic.Field(
        default_factory=list,
        deprecated=deprecated(
            "target_keys is deprecated. "
            "Use update_keys and residual_keys instead."
        ),
        exclude=True,
    )
    """
    Deprecated: List of variable names to be optimized.
    Use update_keys and residual_keys instead.
    """

    convergence_threshold: float = pydantic.Field(0.01, gt=0)
    """
    Convergence threshold for the solver.
    """

    max_iterations: int = pydantic.Field(100, gt=0)
    """
    Maximum number of iterations for the solver.
    """

    divergence_threshold: float = pydantic.Field(100, gt=0)
    """
    Divergence threshold for the solver.
    """

    alpha_component_wise: bool = False

    bb_type: Literal["short", "long"] = "short"
    """
    Type of Barzilai-Borwein method.
    Choose from "short" or "long". Defaults to "short".
    """

    update_keys: list[str] = pydantic.Field(default_factory=list)
    """
    List of variable names to be updated by the solver.
    If empty list is given, update_keys are set to be same as target_keys.
    """

    operator_keys: list[str] = pydantic.Field(default_factory=list)
    """
    List of residual variable names to calculate residuals for.
    If empty list is given, residual values are calculated for
    substracting values of target_keys.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.model_validator(mode="before")
    def pass_target_keys_to_update_and_residual_keys(
        cls, vals: dict[str, Any]
    ) -> dict[str, Any]:
        if not vals.get("target_keys"):
            # None or empty target_keys
            return vals

        target_keys = vals["target_keys"]
        if vals.get("update_keys"):
            raise ValueError(
                "Cannot set both target_keys and update_keys. "
                "target_keys is deprecated. "
                "Use update_keys instead."
            )
        if vals.get("operator_keys"):
            raise ValueError(
                "Cannot set both target_keys and operator_keys. "
                "target_keys is deprecated. "
                "Use operator_keys instead."
            )

        vals["update_keys"] = target_keys
        vals["operator_keys"] = target_keys
        vals.pop("target_keys")

        return vals

    @pydantic.model_validator(mode="after")
    def empty_list_is_not_allowed(self) -> list[str]:
        assert (
            len(self.update_keys) > 0
        ), "Set at least one target item for target_keys in solver settings"

        return self

    @pydantic.model_validator(mode="after")
    def convergence_must_be_less_than_divergence(self) -> Self:
        assert (
            self.convergence_threshold < self.divergence_threshold
        ), "convergence threshold must be less than divergence_threshold"
        return self

    def get_update_keys(self) -> list[str]:
        return self.update_keys


class ConjugateGradientSolverSetting(
    pydantic.BaseModel, IPhlowerIterationSolverSetting
):
    convergence_threshold: float = pydantic.Field(0.01, gt=0)
    """
    Convergence threshold for the solver.
    """

    max_iterations: int = pydantic.Field(100, gt=0)
    """
    Maximum number of iterations for the solver.
    """

    divergence_threshold: float = pydantic.Field(100, gt=0)
    """
    Divergence threshold for the solver.
    """

    update_keys: list[str] = pydantic.Field(default_factory=list)
    """
    List of variable names to be updated by the solver.
    """

    operator_keys: list[str] = pydantic.Field(default_factory=list)
    """
    List of residual variable names to calculate residuals for.
    If empty list is given, residual values are calculated for
    substracting values of update_keys.
    """

    dict_variable_to_right: dict[str, str] = pydantic.Field(
        default_factory=dict
    )
    """
    Dict to map from variable names to names of the corresponding right hand
    side variables. If a item is not found, assume the right hand side is zero
    for that key.
    """

    dict_variable_to_dirichlet: dict[str, str] = pydantic.Field(
        default_factory=dict
    )
    """
    Dict to map from variable names to names of the corresponding dirichlet
    boundary variables. If a item is not found, assume the dirichlet is
    already considered in the operator.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    @pydantic.model_validator(mode="after")
    def empty_list_is_not_allowed(self) -> list[str]:
        assert (
            len(self.update_keys) > 0
        ), "Set at least one update item for update_keys in solver settings"

        return self

    @pydantic.model_validator(mode="after")
    def convergence_must_be_less_than_divergence(self) -> Self:
        assert (
            self.convergence_threshold < self.divergence_threshold
        ), "convergence threshold must be less than divergence_threshold"
        return self

    def get_update_keys(self) -> list[str]:
        return self.update_keys


_name_to_setting: dict[str, IPhlowerIterationSolverSetting] = {
    PhlowerIterationSolverType.none.value: EmptySolverSetting,
    PhlowerIterationSolverType.simple.value: SimpleSolverSetting,
    PhlowerIterationSolverType.bb.value: BarzilaiBoweinSolverSetting,
    PhlowerIterationSolverType.cg.value: ConjugateGradientSolverSetting,
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
