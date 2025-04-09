from __future__ import annotations

import os
import pathlib
from collections.abc import Callable, Iterable
from typing import Any

from phlower.io._file_builder import PhlowerFileBuilder
from phlower.io._files import (
    IPhlowerCheckpointFile,
    IPhlowerNumpyFile,
    IPhlowerYamlFile,
)
from phlower.utils.enums import PhlowerFileExtType


class PhlowerDirectory:
    def __init__(self, path: os.PathLike | PhlowerDirectory) -> None:
        if isinstance(path, PhlowerDirectory):
            self._path = path._path
        else:
            self._path = pathlib.Path(path)

    def __repr__(self) -> str:
        return f"PhlowerDirectory: {self._path}"

    @property
    def path(self) -> pathlib.Path:
        return self._path

    def find_directory(
        self, required_filename: str, recursive: bool = False
    ) -> Iterable[pathlib.Path]:
        yield from self._find_directory(
            self._path, required_filename, recursive
        )

    def _find_directory(
        self, target: pathlib.Path, required_filename: str, recursive: bool
    ) -> Iterable[pathlib.Path]:
        if not target.is_dir():
            return

        for path in target.iterdir():
            if path.is_file():
                if path.name == required_filename:
                    yield target

            if recursive and path.is_dir():
                yield from self._find_directory(
                    path, required_filename, recursive
                )

    def find_yaml_file(
        self, file_base_name: str, *, allow_missing: bool = False
    ) -> IPhlowerYamlFile | None:
        extensions = [
            PhlowerFileExtType.YAML,
            PhlowerFileExtType.YAMLENC,
            PhlowerFileExtType.YML,
            PhlowerFileExtType.YMLENC,
        ]

        return self._find_file(
            file_base_name=file_base_name,
            extensions=extensions,
            builder=PhlowerFileBuilder.yaml_file,
            allow_missing=allow_missing,
        )

    def find_variable_file(
        self, variable_name: str, *, allow_missing: bool = False
    ) -> IPhlowerNumpyFile | None:
        extensions = [
            PhlowerFileExtType.NPY,
            PhlowerFileExtType.NPYENC,
            PhlowerFileExtType.NPZ,
            PhlowerFileExtType.NPZENC,
        ]
        return self._find_file(
            variable_name,
            extensions,
            PhlowerFileBuilder.numpy_file,
            allow_missing=allow_missing,
        )

    def _find_file(
        self,
        file_base_name: str,
        extensions: list[PhlowerFileExtType],
        builder: Callable[[pathlib.Path], Any] | None = None,
        *,
        allow_missing: bool = False,
    ) -> pathlib.Path:
        if builder is None:
            builder = lambda x: x  # NOQA

        for ext in extensions:
            path: pathlib.Path = self._path / (file_base_name + ext.value)
            if path.exists():
                return builder(path)

        if allow_missing:
            return None

        raise ValueError(
            f"Unknown extension or file not found for {file_base_name}"
            f" in {self.path}"
        )

    def exist_variable_file(self, variable_name: str) -> bool:
        _file = self.find_variable_file(variable_name, allow_missing=True)
        return _file is not None

    def find_snapshot_files(self) -> list[IPhlowerCheckpointFile]:
        snapshots = [
            PhlowerFileBuilder.checkpoint_file(p)
            for p in list(self._path.glob("**/snapshot_epoch_*"))
        ]
        return snapshots

    def find_csv_file(
        self, file_base_name: str, *, allow_missing: bool = False
    ) -> pathlib.Path | None:
        return self._find_file(
            file_base_name,
            extensions=[PhlowerFileExtType.CSV],
            allow_missing=allow_missing,
        )
