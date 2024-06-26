import abc
import pathlib

import numpy as np

from phlower._base.array import IPhlowerArray
from phlower.utils.typing import ArrayDataType


class IPhlowerBaseFile(metaclass=abc.ABCMeta):
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
    @abc.abstractclassmethod
    def save_variables(
        cls,
        output_directory: pathlib.Path,
        file_basename: str,
        data: ArrayDataType,
        *,
        dtype: type = np.float32,
        encrypt_key: bytes = None,
    ): ...

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
    def save(
        self,
        data: ArrayDataType,
        *,
        encrypt_key: bytes = None,
        overwrite: bool = True,
    ) -> None: ...


class IPhlowerPickleFile(IPhlowerBaseFile, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(self, *, decrypt_key: bytes = None) -> dict:
        raise NotImplementedError()

    @abc.abstractmethod
    def save(
        self,
        dump_data: object,
        overwrite: bool = False,
        encrypt_key: bytes = None,
    ) -> None:
        raise NotImplementedError()


class IPhlowerYamlFile(IPhlowerBaseFile, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def load(self, *, decrypt_key: bytes = None) -> dict:
        raise NotImplementedError()

    @abc.abstractmethod
    def save(
        self,
        dump_data: object,
        overwrite: bool = False,
        encrypt_key: bytes = None,
    ) -> None:
        raise NotImplementedError()


class IPhlowerCheckpointFile(IPhlowerBaseFile, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def epoch(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, device: str, *, decrypt_key: bytes = None) -> dict:
        raise NotImplementedError()

    @abc.abstractmethod
    def save(
        self,
        dump_data: object,
        overwrite: bool = False,
        encrypt_key: bytes = None,
    ) -> None:
        raise NotImplementedError()
