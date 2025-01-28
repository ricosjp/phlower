from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName
from phlower.utils.preprocess import convert_to_dumped
from phlower.utils.typing import ArrayDataType


class IsoAMScaler(BaseEstimator, TransformerMixin, IPhlowerScaler):
    """Class to perform scaling for IsoAM based on
    https://arxiv.org/abs/2005.06316.
    """

    @classmethod
    def create(cls, name: str, **kwards) -> IsoAMScaler:
        if name == PhlowerScalerName.ISOAM_SCALE.value:
            return IsoAMScaler(**kwards)

        raise NotImplementedError()

    def __init__(self, other_components: list[str], **kwargs):
        self.var_: float = 0.0
        self.std_: float = 0.0
        self.mean_square_: float = 0.0
        self.n_: int = 0
        self.other_components = other_components

        if len(self.other_components) == 0:
            raise ValueError(
                "To use IsoAMScaler, feed other_components: "
                f"{self.other_components}"
            )

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def component_dim(self) -> int:
        # num. of others + self
        return len(self.other_components) + 1

    @property
    def use_diagonal(self) -> bool:
        return True

    def is_fitted(self) -> bool:
        return self.n_ != 0

    def is_erroneous(self) -> bool:
        return np.any(np.isnan(self.var_))

    def partial_fit(self, data: ArrayDataType) -> None:
        if len(data.shape) != 1:
            raise ValueError(f"Input data should be 1D: {data}")

        m = len(data)
        mean_square = (self.mean_square_ * self.n_ + np.sum(data**2)) / (
            self.n_ + m
        )

        self.mean_square_ = mean_square
        self.n_ += m

        # To use mean_i [x_i^2 + y_i^2 + z_i^2], multiply by the dim
        # because self.n_ = self.component_dim * data_size
        self.var_ = self.mean_square_ * self.component_dim
        self.std_ = np.sqrt(self.var_)
        return

    def transform(self, data: ArrayDataType) -> ArrayDataType:
        if self.std_ == 0.0:
            raise ValueError("std value is 0.")

        return data * (1.0 / self.std_)

    def inverse_transform(self, data: ArrayDataType) -> ArrayDataType:
        return data * self.std_

    def get_dumped_data(self) -> dict[str, Any]:
        return {k: convert_to_dumped(v) for k, v in vars(self).items()}
