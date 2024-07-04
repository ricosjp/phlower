from __future__ import annotations

import abc

import numpy as np

from phlower.utils.typing import ArrayDataType


class IPhlowerScaler(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def create(cls, name: str, **kwards) -> IPhlowerScaler: ...

    @property
    @abc.abstractclassmethod
    def get_registered_names(cls) -> list[str]: ...

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
