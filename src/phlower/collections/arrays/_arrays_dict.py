import abc
from collections.abc import Sequence

from phlower._base.array import IPhlowerArray
from phlower.utils.typing import ArrayDataType


class IPhlowerDictArray(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def is_sparse(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def __getitem__(self, key: str) -> ArrayDataType:
        raise NotImplementedError()

    @abc.abstractmethod
    def concatenate(self, key: str) -> ArrayDataType:
        raise NotImplementedError()


class SequencedDictArray:
    def __init__(self, data: list[dict[str, IPhlowerArray]]) -> None:
        self._data = data

    def get_names(self) -> Sequence[str]:
        return self._data[0].keys()

    def concatenate(self) -> dict[str, IPhlowerArray]:
        return {name: self._concatenate(name) for name in self.get_names()}

    def _concatenate(self, name: str) -> IPhlowerArray:
        assert len(self._data) == 1
        return self._data[0][name]
        # TODO: Under construction
        # change how to concatenate ndarray
        # according to properties such as time_series, sparse
        # return np.concatenate([v[name] for v in self._data])
