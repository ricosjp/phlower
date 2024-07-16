from __future__ import annotations

from phlower.utils.enums import PhysicalDimensionType
from phlower.utils.exceptions import InvalidDimensionError


class PhysicsDimensions:
    def __init__(self, dimensions: dict[str, float]) -> None:
        self._check(dimensions)

        self._dimensions = {
            name: 0 for name in PhysicalDimensionType.__members__
        }
        self._dimensions |= dimensions

    def _check(self, dimensions: dict[str, float]):
        for name in dimensions:
            if not PhysicalDimensionType.is_exist(name):
                raise InvalidDimensionError(
                    f"{name} is not valid dimension name."
                )

    def get(self, name: str) -> float | None:
        return self._dimensions.get(name)

    def __getitem__(self, name: str) -> float:
        if not PhysicalDimensionType.is_exist(name):
            raise InvalidDimensionError(f"{name} is not valid dimension name.")
        return self._dimensions[name]

    def __eq__(self, other: PhysicsDimensions):
        for k in PhysicalDimensionType.__members__:
            if self._dimensions[k] != other[k]:
                return False

        return True

    def to_list(self) -> list[float]:
        _list: list[float] = [0 for _ in range(len(PhysicalDimensionType))]
        for k, v in self._dimensions.items():
            if k not in PhysicalDimensionType.__members__:
                raise NotImplementedError(
                    f"dimension name: {k} is not implemented."
                    f"Avaliable units are {list(PhysicalDimensionType)}"
                )
            _list[PhysicalDimensionType[k].value] = v

        return _list
