from __future__ import annotations

import abc
from typing import Any

import numpy as np

from phlower.utils.typing import ArrayDataType


class IPhlowerScaler(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def create(cls, name: str, **kwards) -> IPhlowerScaler: ...

    @property
    @abc.abstractmethod
    def use_diagonal(self) -> bool: ...

    @abc.abstractmethod
    def is_erroneous(self) -> bool: ...

    @abc.abstractmethod
    def transform(self, data: ArrayDataType) -> np.ndarray: ...

    @abc.abstractmethod
    def partial_fit(self, data: ArrayDataType) -> None: ...

    @abc.abstractmethod
    def inverse_transform(self, data: ArrayDataType) -> np.ndarray: ...

    @abc.abstractmethod
    def get_dumped_data(self) -> dict[str, Any]: ...
