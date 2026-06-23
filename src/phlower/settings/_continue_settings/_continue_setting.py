from __future__ import annotations

from typing import Annotated, Any

import pydantic
from pydantic import (
    PlainSerializer,
    PlainValidator,
    SerializationInfo,
    ValidationInfo,
)

from phlower.utils.enums import ContinueTrainingType

from ._empty_parameters import EmptyContinueSetting
from ._interface import IReadOnlyContinueSetting
from ._lr_decay_parameters import LRDecayContinueSetting
from ._optimizer_switch_setting import OptimizerSwitchContinueSetting


def _validate(value: object, info: ValidationInfo) -> str:
    if "continue_type" not in info.data:
        raise ValueError(f"continue_type is not defined in {info.data}")

    continue_type: ContinueTrainingType | str = info.data["continue_type"]

    match continue_type:
        case ContinueTrainingType.none:
            return EmptyContinueSetting.model_validate(value)
        case ContinueTrainingType.lr_decay:
            return LRDecayContinueSetting.model_validate(value)
        case ContinueTrainingType.optimizer_switch:
            return OptimizerSwitchContinueSetting.model_validate(value)
        case _:
            raise ValueError(
                f"Invalid continue type value is set. Input: {continue_type}"
            )


def _serialize(
    v: pydantic.BaseModel, info: SerializationInfo
) -> dict[str, Any]:
    return v.model_dump()


ContinueParameterSettingType = Annotated[
    EmptyContinueSetting
    | LRDecayContinueSetting
    | OptimizerSwitchContinueSetting,
    PlainValidator(_validate),
    PlainSerializer(_serialize),
]


class ContinueSetting(pydantic.BaseModel, IReadOnlyContinueSetting):
    is_active: bool = True

    stop_count: int = 10
    """
    Number of times to continue training.
    Training will be stopped after `stop_count` times.
    """

    continue_type: ContinueTrainingType = ContinueTrainingType.none
    """
    Name of the continue training type.
    """

    parameters: ContinueParameterSettingType = pydantic.Field(
        default_factory=dict
    )
    """
    Parameters for the continue training type.
    This content depends on the value of `continue_type`.
    """

    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
        validate_default=True,
    )

    def get_stop_count(self) -> int:
        return self.stop_count

    @pydantic.field_serializer("continue_type")
    @classmethod
    def serialize_continue_type(cls, value: ContinueTrainingType) -> str:
        return value.value

    @pydantic.model_validator(mode="after")
    def validate_parameters(self) -> ContinueSetting:
        self.parameters.validate(self)
        return self
