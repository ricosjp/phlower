from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName
from phlower.utils.typing import ArrayDataType


class LogitTransformScaler(BaseEstimator, TransformerMixin, IPhlowerScaler):
    """Class to perform identity conversion (do nothing)."""

    @classmethod
    def create(cls, name: str, **kwards) -> LogitTransformScaler:
        if name == PhlowerScalerName.logit_transform.value:
            return LogitTransformScaler(**kwards)

        raise NotImplementedError()

    def __init__(
        self, upper_bound: float = 0.99, lower_bound: float = 0.01, **kwards
    ) -> None:
        super().__init__()
        self._upper_bound = upper_bound
        self._lower_bound = lower_bound
        assert 0.0 < self._lower_bound < self._upper_bound < 1.0, (
            "LogitTransformScaler requires 0.0 < lower_bound"
            " < upper_bound < 1.0"
        )

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
        data = np.clip(data, self._lower_bound, self._upper_bound)
        return np.log(data / (1.0 - data))

    def inverse_transform(self, data: ArrayDataType) -> ArrayDataType:
        return 1.0 / (1.0 + np.exp(-data))

    def get_dumped_data(self) -> dict[str, str | int | float]:
        return {
            "upper_bound": self._upper_bound,
            "lower_bound": self._lower_bound,
        }
