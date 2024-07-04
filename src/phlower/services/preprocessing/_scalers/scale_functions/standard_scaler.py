from __future__ import annotations

import numpy as np
from sklearn import preprocessing

from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName


class StandardScaler(preprocessing.StandardScaler, IPhlowerScaler):
    @classmethod
    def create(cls, name: str, **kwards) -> StandardScaler:
        if name == PhlowerScalerName.STANDARDIZE.value:
            with_mean = kwards.pop("with_mean", True)
            if not with_mean:
                raise ValueError("with_mean must be True in standardize scaler")
            return StandardScaler(with_mean=with_mean, **kwards)

        if name == PhlowerScalerName.STD_SCALE.value:
            with_mean = kwards.pop("with_mean", False)
            if with_mean:
                raise ValueError("with_mean must be False in std_scale")
            return StandardScaler(with_mean=with_mean, **kwards)

        raise NotImplementedError(
            f"Instance for {name} is not implemented in {cls.__class__}"
        )

    @classmethod
    def get_registered_names(self) -> list[str]:
        return [
            PhlowerScalerName.STANDARDIZE.value,
            PhlowerScalerName.STD_SCALE.value,
        ]

    def __init__(
        self,
        *,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
        **kwargs,
    ):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return np.any(np.isnan(self.var_))
