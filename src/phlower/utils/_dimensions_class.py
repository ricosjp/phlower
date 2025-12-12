from __future__ import annotations

from typing import Annotated

from phlower_tensor import PhysicalDimensions
from pydantic import (
    PlainSerializer,
    PlainValidator,
    SerializationInfo,
    ValidationInfo,
)


def _validate(v: dict, info: ValidationInfo) -> PhysicalDimensions:
    if not isinstance(v, dict):
        raise TypeError(f"Expected dictionary, but got {type(v)}")

    try:
        return PhysicalDimensions(v)
    except Exception as ex:
        raise TypeError("Validation for physical dimension is failed.") from ex


def _serialize(
    v: PhysicalDimensions, info: SerializationInfo
) -> dict[str, float]:
    return v.to_dict()


PhysicalDimensionsClass = Annotated[
    PhysicalDimensions,
    PlainValidator(_validate),
    PlainSerializer(_serialize),
]
