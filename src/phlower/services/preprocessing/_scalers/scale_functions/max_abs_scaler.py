from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName
from phlower.utils.typing import ArrayDataType


class MaxAbsPoweredScaler(BaseEstimator, TransformerMixin, IPhlowerScaler):
    @classmethod
    def create(cls, name: str, **kwards) -> MaxAbsPoweredScaler:
        if name == PhlowerScalerName.MAX_ABS_POWERED.value:
            return MaxAbsPoweredScaler(**kwards)

        raise NotImplementedError(
            f"Instance for {name} is not implemented in {cls.__class__}"
        )

    def __init__(self, power: float = 1.0, **kwargs):
        self.max_: NDArray | None = None
        self.power = power

        for k, v in kwargs.items():
            setattr(self, k, self._convert(k, v))

    def _convert(self, field_name: str, value: float | None) -> float:
        if field_name == "max_":
            if value is None:
                return value
            return np.array(value)

        if field_name == "power":
            return float(value)

        raise NotImplementedError(f"Unknown field name : {field_name}")

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return False

    def partial_fit(self, data: ArrayDataType) -> None:
        if sp.issparse(data):
            _max = np.ravel(np.max(np.abs(data), axis=0).toarray())
        else:
            _max = np.max(np.abs(data), axis=0)

        if self.max_ is None:
            self.max_ = _max
        else:
            self.max_ = np.maximum(_max, self.max_)

    def is_fitted(self) -> bool:
        return self.max_ is not None

    def transform(self, data: ArrayDataType) -> ArrayDataType:
        if not self.is_fitted():
            raise ValueError(
                f"This scaler has not fitted yet. {self.__class__.__name__}"
            )

        if np.max(self.max_) == 0.0:
            raise ValueError(
                "Transform cannot be performed because one of max values is 0."
            )

        if sp.issparse(data):
            if len(self.max_) != 1:
                raise ValueError("Should be componentwise: false")
            return data * ((1.0 / self.max_[0]) ** self.power)

        return data * ((1.0 / self.max_) ** self.power)

    def inverse_transform(self, data: ArrayDataType) -> ArrayDataType:
        if not self.is_fitted():
            raise ValueError(
                f"This scaler has not fitted yet. {self.__class__.__name__}"
            )

        if sp.issparse(data):
            if len(self.max_) != 1:
                raise ValueError("Should be componentwise: false")
            inverse_scale = self.max_[0] ** (self.power)
            return data * inverse_scale

        return data * (self.max_**self.power)

    def get_dumped_data(self) -> dict[str, str | int | float]:
        return {
            "max_": None if self.max_ is None else self.max_.tolist(),
            "power": self.power,
        }
