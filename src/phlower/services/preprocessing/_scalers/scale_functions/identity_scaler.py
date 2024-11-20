from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin

from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName
from phlower.utils.typing import ArrayDataType


class IdentityScaler(BaseEstimator, TransformerMixin, IPhlowerScaler):
    """Class to perform identity conversion (do nothing)."""

    @classmethod
    def create(cls, name: str, **kwards) -> IdentityScaler:
        if name == PhlowerScalerName.IDENTITY.value:
            return IdentityScaler(**kwards)

        raise NotImplementedError()

    def __init__(self, **kwards) -> None:
        super().__init__()

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
        return data

    def inverse_transform(self, data: ArrayDataType) -> ArrayDataType:
        return data

    def get_dumped_data(self) -> dict[str, str | int | float]:
        return {}
