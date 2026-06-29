from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple

from sklearn.base import BaseEstimator, TransformerMixin

from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName
from phlower.utils.typing import ArrayDataType


class PhysicalDataType(StrEnum):
    velocity = "velocity"
    pressure = "pressure"
    length = "length"
    pressure_by_density = "pressure_by_density"


class _CharacteristicValues(NamedTuple):
    rho: float
    nu: float
    U: float
    L: float

    def to_dict(self) -> dict[str, float]:
        return {
            "rho": self.rho,
            "nu": self.nu,
            "U": self.U,
            "L": self.L,
        }

    def get_divisor(self, data_type: PhysicalDataType) -> float:
        match data_type:
            case PhysicalDataType.velocity:
                return self.U
            case PhysicalDataType.pressure:
                return self.rho * self.U**2
            case PhysicalDataType.length:
                return self.L
            case PhysicalDataType.pressure_by_density:
                return self.U**2
            case _:
                raise NotImplementedError(f"Invalid data_type: {data_type}")


class PhysicalNondimensionalizeScaler(
    BaseEstimator, TransformerMixin, IPhlowerScaler
):
    """Class to perform identity conversion (do nothing)."""

    @classmethod
    def create(cls, name: str, **kwards) -> PhysicalNondimensionalizeScaler:
        if name == PhlowerScalerName.physical_nondimensionalize:
            return PhysicalNondimensionalizeScaler(**kwards)

        raise NotImplementedError()

    def __init__(
        self,
        data_type: PhysicalDataType | str,
        rho: float,
        nu: float,
        U: float,
        L: float,
    ) -> None:
        super().__init__()

        if isinstance(data_type, str):
            data_type = PhysicalDataType(data_type)
        self._data_type = data_type
        self._const = _CharacteristicValues(rho=rho, nu=nu, U=U, L=L)

        if data_type not in PhysicalDataType:
            raise ValueError(f"Invalid data_type: {data_type}")

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_fitted(self) -> bool:
        return True

    def is_erroneous(self) -> bool:
        return False

    def partial_fit(self, data: ArrayDataType) -> None:
        return

    def transform(self, data: ArrayDataType) -> ArrayDataType:
        divisor = self._const.get_divisor(self._data_type)
        return data / divisor

    def inverse_transform(self, data: ArrayDataType) -> ArrayDataType:
        return data * self._const.get_divisor(self._data_type)

    def get_dumped_data(self) -> dict[str, str | int | float]:
        return {
            "data_type": self._data_type.value,
            **self._const.to_dict(),
        }
