from __future__ import annotations

import abc
import pathlib

from phlower._base import IPhlowerArray


class IPhlowerBaseFile(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def save(
        cls,
        output_directory: pathlib.Path,
        file_basename: str,
        data: object,
        *,
        encrypt_key: bytes = None,
        allow_overwrite: bool = False,
        **kwargs,
    ) -> IPhlowerBaseFile: ...

    @abc.abstractmethod
    def __init__(self, path: pathlib.Path) -> None:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def is_encrypted(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def file_path(self) -> pathlib.Path:
        raise NotImplementedError()


class IPhlowerNumpyFile(IPhlowerBaseFile, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def file_extension(self) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(
        self, *, check_nan: bool = False, decrypt_key: bytes = None
    ) -> IPhlowerArray:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_variable_name(self) -> str:
        raise NotImplementedError()


class IPhlowerYamlFile(IPhlowerBaseFile, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(self, *, decrypt_key: bytes = None) -> dict:
        raise NotImplementedError()


class IPhlowerCheckpointFile(IPhlowerBaseFile, metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def get_fixed_prefix(self) -> int: ...

    @property
    @abc.abstractmethod
    def epoch(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(
        self, device: str | None = None, *, decrypt_key: bytes = None
    ) -> dict:
        raise NotImplementedError()


def to_pathlib_object(
    path: pathlib.Path | str | IPhlowerBaseFile,
) -> pathlib.Path:
    if isinstance(path, pathlib.Path):
        return path
    if isinstance(path, IPhlowerYamlFile):
        return path.file_path
    if isinstance(path, str):
        return pathlib.Path(path)

    raise NotImplementedError(f"{path} cannot be converted to pathlib.Path")
