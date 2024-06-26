import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName


class MaxAbsScaler(BaseEstimator, TransformerMixin, IPhlowerScaler):

    @classmethod
    def create(cls, name: str, **kwards):
        if name == PhlowerScalerName.MAX_ABS.name:
            return MaxAbsScaler(**kwards)

        raise NotImplementedError()

    @classmethod
    def get_registered_names(self) -> list[str]:
        return [PhlowerScalerName.MAX_ABS.name]

    def __init__(self, power=1.0, **kwargs):
        self.max_ = 0.0
        self.power = power
        return

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return False

    def partial_fit(self, data):
        if sp.issparse(data):
            self.max_ = np.maximum(
                np.ravel(np.max(np.abs(data), axis=0).toarray()), self.max_
            )
        else:
            self.max_ = np.maximum(np.max(np.abs(data), axis=0), self.max_)
        return self

    def transform(self, data):
        if np.max(self.max_) == 0.0:
            scale = 0.0
        else:
            scale = (1 / self.max_) ** self.power

        if sp.issparse(data):
            if len(scale) != 1:
                raise ValueError("Should be componentwise: false")
            scale = scale[0]
        return data * scale

    def inverse_transform(self, data):
        inverse_scale = self.max_
        if sp.issparse(data):
            if len(inverse_scale) != 1:
                raise ValueError("Should be componentwise: false")
            inverse_scale = inverse_scale[0] ** (self.power)
        return data * inverse_scale
