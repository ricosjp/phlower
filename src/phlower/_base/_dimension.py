from __future__ import annotations

from typing import Annotated

from pydantic import (
    PlainSerializer,
    PlainValidator,
    SerializationInfo,
    ValidationInfo,
)

from phlower.utils.enums import PhysicalDimensionSymbolType
from phlower.utils.exceptions import InvalidDimensionError


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


class PhysicalDimensions:
    def __init__(self, dimensions: dict[str, float]) -> None:
        self._check(dimensions)

        self._dimensions = {
            name: 0 for name in PhysicalDimensionSymbolType.__members__
        }
        self._dimensions |= dimensions

    def __eq__(self, other: PhysicalDimensions) -> bool:
        for k in PhysicalDimensionSymbolType.__members__.keys():
            if self._dimensions[k] != other[k]:
                return False

        return True

    def _check(self, dimensions: dict[str, float]):
        for name in dimensions:
            if not PhysicalDimensionSymbolType.is_exist(name):
                raise InvalidDimensionError(
                    f"dimension name: {name} is not implemented."
                    f"Avaliable units are {list(PhysicalDimensionSymbolType)}"
                )
            if dimensions[name] is None:
                raise InvalidDimensionError(
                    f"None dimension is found in {name}"
                )

    def get(self, name: str) -> float | None:
        return self._dimensions.get(name)

    def __getitem__(self, name: str) -> float:
        if not PhysicalDimensionSymbolType.is_exist(name):
            raise InvalidDimensionError(f"{name} is not valid dimension name.")
        return self._dimensions[name]

    def to_list(self) -> list[float]:
        _list: list[float] = [0 for _ in PhysicalDimensionSymbolType]
        for k, v in self._dimensions.items():
            if k not in PhysicalDimensionSymbolType.__members__:
                raise InvalidDimensionError(
                    f"dimension name: {k} is not implemented."
                    f"Avaliable units are {list(PhysicalDimensionSymbolType)}"
                )
            _list[PhysicalDimensionSymbolType[k].value] = v

        return _list

    def to_dict(self) -> dict:
        return self._dimensions


PhysicalDimensionsClass = Annotated[
    PhysicalDimensions,
    PlainValidator(_validate),
    PlainSerializer(_serialize),
]
