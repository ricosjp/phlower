from collections.abc import Sequence

from phlower.base.array import IPhlowerArray, phlower_arrray
from phlower.utils.typing import ArrayDataType

from ._interface import IPhlowerArrayVariables


class SequencedDictDenseArray:
    def __init__(self, data: list[dict[str, IPhlowerArray]]) -> None:
        self._data = data

    def get_names(self) -> Sequence[str]:
        return self._data[0].keys()

    def concatenate(self):
        return {name: self._concatenate(name) for name in self.get_names()}

    def _concatenate(self, name: str) -> ArrayDataType:
        assert len(self._data) == 1
        return self._data[0][name]
        # TODO: Under construction
        # return np.concatenate([v[name] for v in self._data])
