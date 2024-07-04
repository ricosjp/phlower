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

        raise NotImplementedError()

    @classmethod
    def get_registered_names(self) -> list[str]:
        return [PhlowerScalerName.MAX_ABS_POWERED.value]

    def __init__(self, power: float = 1.0, **kwargs):
        self._max: NDArray | None = None
        self.power = power
        return

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

        if self._max is None:
            self._max = _max
        else:
            self._max = np.maximum(_max, self._max)

    def is_fitted(self) -> bool:
        return self._max is not None

    def transform(self, data: ArrayDataType):
        if not self.is_fitted():
            raise ValueError(
                f"This scaler has not fitted yet. {self.__class__.__name__}"
            )

        if np.max(self._max) == 0.0:
            raise ValueError(
                "Transform cannot be performed because one of max values is 0."
            )

        if sp.issparse(data):
            if len(self._max) != 1:
                raise ValueError("Should be componentwise: false")
            return data * ((1.0 / self._max[0]) ** self.power)

        return data * ((1.0 / self._max) ** self.power)

    def inverse_transform(self, data: ArrayDataType):
        if not self.is_fitted():
            raise ValueError(
                f"This scaler has not fitted yet. {self.__class__.__name__}"
            )

        if sp.issparse(data):
            if len(self._max) != 1:
                raise ValueError("Should be componentwise: false")
            inverse_scale = self._max[0] ** (self.power)
            return data * inverse_scale

        return data * (self._max**self.power)
