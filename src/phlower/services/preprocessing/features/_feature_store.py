from __future__ import annotations

import pathlib
from collections.abc import Iterable
from typing import get_args

import pydantic.dataclasses as dc

from phlower.io._files import PhlowerNumpyFile
from phlower.utils.exceptions import PhlowerFeatureStoreOverwriteError
from phlower.utils.typing import ArrayDataType, SparseArrayType

# TODO: maybe it is covenient if DataStore support pipe expression
# Implementing __iter__ is only necessary ??


class FeatureDataStore:
    def __init__(
        self, feature_table: dict[str, ArrayDataType | FeatureDataVO]
    ) -> None:
        self._feature_table: dict[str, FeatureDataVO] = {}
        for name, v in feature_table.items():
            if isinstance(v, FeatureDataVO):
                self._feature_table[name] = v
                continue

            if isinstance(v, get_args(ArrayDataType)):
                self._feature_table[name] = FeatureDataVO(name, v)
                continue

            raise NotImplementedError(
                f"Unexpected type is found in {name} : {type(v)}"
            )

    def __getitem__(self, name: str) -> FeatureDataVO:
        return self._feature_table[name]

    def register(self, name: str, value: ArrayDataType) -> None:
        self._check_duplicate(name)
        data = FeatureDataVO(name=name, data=value)
        self._feature_table[name] = data

    def _check_duplicate(self, name: str) -> None:
        if name in self._feature_table.keys():
            raise PhlowerFeatureStoreOverwriteError(
                f"Key name: {name} has already existed."
                " Overwriting item is forbidden."
            )
        return

    def keys(self) -> Iterable[str]:
        return self._feature_table.keys()

    def values(self) -> Iterable[ArrayDataType]:
        return self._feature_table.values()

    def save(
        self, output_directory: pathlib.Path, encrypt_key: bytes | None = None
    ) -> None:
        for item in self._feature_table.values():
            PhlowerNumpyFile.save_variables(
                output_directory=output_directory,
                file_basename=item.name,
                data=item.data,
                dtype=item.data.dtype,
                encrypt_key=encrypt_key,
            )


@dc.dataclass(frozen=True)
class FeatureDataVO:
    name: str
    data: ArrayDataType

    @property
    def is_sparse(self) -> bool:
        return isinstance(self.data, get_args(SparseArrayType))
