from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import preprocessing

from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName
from phlower.utils.preprocess import convert_to_dumped


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

    def __init__(
        self,
        *,
        copy: bool = True,
        with_mean: bool = True,
        with_std: bool = True,
        **kwargs,
    ):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

        for k, v in kwargs.items():
            setattr(self, k, self._convert(k, v))

    def _convert(self, field_name: str, value: Any) -> Any:  # noqa: ANN401
        if field_name in ["copy", "with_mean", "with_std"]:
            return value

        if field_name.endswith("_"):
            if isinstance(value, list):
                return np.array(value)

            if isinstance(value, int | float):
                return value

        raise NotImplementedError(
            f"{field_name} is not understandable. value: {value}"
        )

    @property
    def use_diagonal(self) -> bool:
        return False

    def is_erroneous(self) -> bool:
        return np.any(np.isnan(self.var_))

    def get_dumped_data(self) -> dict[str, Any]:
        return {k: convert_to_dumped(v) for k, v in vars(self).items()}
