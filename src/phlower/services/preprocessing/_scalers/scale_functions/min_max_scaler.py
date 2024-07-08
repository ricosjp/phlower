from typing import Any, get_args

import numpy as np
from numpy.typing import NDArray
from sklearn import preprocessing

from phlower.services.preprocessing._scalers import IPhlowerScaler
from phlower.utils.enums import PhlowerScalerName
from phlower.utils.preprocess import convert_to_dumped


class MinMaxScaler(preprocessing.MinMaxScaler, IPhlowerScaler):
    @classmethod
    def create(cls, name: str, **kwards):
        if name == PhlowerScalerName.MIN_MAX.value:
            return MinMaxScaler(**kwards)

        raise NotImplementedError()

    @classmethod
    def get_registered_names(self) -> list[str]:
        return [PhlowerScalerName.MIN_MAX.value]

    def __init__(
        self, feature_range=(0, 1), *, copy=True, clip=False, **kwargs
    ):
        super().__init__(feature_range, copy=copy, clip=clip)
        for k, v in kwargs.items():
            setattr(self, k, self._convert(k, v))

    def _convert(self, field_name: str, value: Any):
        if field_name in "feature_range":
            return tuple(value)

        if field_name in ["copy", "clip"]:
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
        return False

    def get_dumped_data(self) -> dict[str, Any]:
        return {k: convert_to_dumped(v) for k, v in vars(self).items()}
