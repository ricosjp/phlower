import abc
from collections.abc import Sequence

import numpy as np

from phlower.utils.typing import ArrayDataType, DenseArrayType, SparseArrayType


class IConcatenator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def merge(self, batch: Sequence[ArrayDataType]) -> ArrayDataType:
        raise NotImplementedError()


class DenseArrayCocatenator(IConcatenator):
    def __init__(self) -> None:
        super().__init__()

    def merge(self, batch: Sequence[DenseArrayType]) -> DenseArrayType:
        np.concatenate(batch)
        return super().merge(batch)


class SparseArrayConcatenator(IConcatenator):
    def __init__(self) -> None:
        super().__init__()

    def merge(self, batch: Sequence[SparseArrayType]) -> SparseArrayType:
        return
