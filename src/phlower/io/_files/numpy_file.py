from __future__ import annotations

import abc
import io
import os
import pathlib
from typing import Any, get_args

import numpy as np
import scipy.sparse as sp

from phlower import utils
from phlower._base.array import IPhlowerArray, phlower_array
from phlower.io._files._interface import IPhlowerNumpyFile, to_pathlib_object
from phlower.utils import get_logger
from phlower.utils.enums import PhlowerFileExtType
from phlower.utils.typing import ArrayDataType, SparseArrayType

_logger = get_logger(__name__)


class PhlowerNumpyFile(IPhlowerNumpyFile):
    @classmethod
    def save(
        cls,
        output_directory: pathlib.Path,
        file_basename: str,
        data: object,
        *,
        encrypt_key: bytes = None,
        allow_overwrite: bool = False,
        dtype: np.dtype = np.float32,
        **kwargs,
    ) -> PhlowerNumpyFile:
        fileio = _get_fileio(data, encrypt_key=encrypt_key)
        save_path = fileio.get_save_path(output_directory, file_basename)
        if not allow_overwrite:
            if save_path.exists():
                raise FileExistsError(f"{save_path} already exists.")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        fileio.save(
            save_path=save_path, data=data, dtype=dtype, encrypt_key=encrypt_key
        )
        _logger.info(f"{file_basename} is saved in: {save_path}")
        return PhlowerNumpyFile(save_path)

    def __init__(self, path: os.PathLike | IPhlowerNumpyFile) -> None:
        path = to_pathlib_object(path)
        ext = self._check_extension_type(path)
        self._path = path
        self._ext_type = ext
        self._file_io = self._get_fileio(ext)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self._path}"

    def _check_extension_type(self, path: pathlib.Path) -> PhlowerFileExtType:
        extensions = [
            PhlowerFileExtType.NPY,
            PhlowerFileExtType.NPYENC,
            PhlowerFileExtType.NPZ,
            PhlowerFileExtType.NPZENC,
        ]
        for ext in extensions:
            if path.name.endswith(ext.value):
                return ext

        raise NotImplementedError(f"Unknown file extension: {path}")

    def _get_fileio(self, ext: PhlowerFileExtType) -> INumpyFileIOCore:
        if ext == PhlowerFileExtType.NPY:
            return _NpyFileIO

        if ext == PhlowerFileExtType.NPYENC:
            return _NpyEncFileIO

        if ext == PhlowerFileExtType.NPZ:
            return _NpzFileIO

        if ext == PhlowerFileExtType.NPZENC:
            return _NpzEncFileIO

        raise NotImplementedError(f"{ext} is not implemented")

    @property
    def name(self) -> str:
        return self.file_path.name

    @property
    def is_encrypted(self) -> bool:
        if self._ext_type == PhlowerFileExtType.NPYENC:
            return True
        if self._ext_type == PhlowerFileExtType.NPZENC:
            return True

        return False

    @property
    def file_extension(self) -> str:
        return self._ext_type.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self, *, check_nan: bool = False, decrypt_key: bytes = None
    ) -> IPhlowerArray:
        loaded_data = self._file_io.load(self._path, decrypt_key=decrypt_key)
        if check_nan and np.any(np.isnan(loaded_data)):
            raise ValueError(f"NaN found in {self._path}")

        return phlower_array(loaded_data)

    def get_variable_name(self) -> str:
        ext = self.file_extension
        name = self._path.name.removesuffix(ext)
        return name


def _get_fileio(
    data: ArrayDataType, encrypt_key: bytes | None = None
) -> INumpyFileIOCore:
    if isinstance(data, np.ndarray):
        if encrypt_key is not None:
            return _NpyEncFileIO()

        return _NpyFileIO()

    if isinstance(data, get_args(SparseArrayType)):
        if encrypt_key is not None:
            return _NpzEncFileIO()

        return _NpzFileIO()

    raise NotImplementedError(f"File IO for {type(data)} is not implemented.")


class INumpyFileIOCore(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def load(cls, path: pathlib.Path, decrypt_key: bytes = None): ...

    @classmethod
    @abc.abstractmethod
    def get_save_path(
        cls, output_directory: pathlib.Path, file_basename: str
    ) -> pathlib.Path: ...

    @classmethod
    @abc.abstractmethod
    def save(
        cls,
        save_path: pathlib.Path,
        data: ArrayDataType,
        dtype: np.dtype = np.float32,
        encrypt_key: bytes = None,
    ) -> None: ...


class _NpyFileIO(INumpyFileIOCore):
    @classmethod
    def load(
        cls, path: pathlib.Path, decrypt_key: bytes = None
    ) -> ArrayDataType:
        return np.load(path)

    @classmethod
    def get_save_path(
        cls, output_directory: pathlib.Path, file_basename: str
    ) -> pathlib.Path:
        return output_directory / (file_basename + PhlowerFileExtType.NPY.value)

    @classmethod
    def save(
        cls,
        save_path: pathlib.Path,
        data: ArrayDataType,
        dtype: np.dtype = np.float32,
        encrypt_key: bytes = None,
    ) -> None:
        np.save(save_path, data.astype(dtype))
        return save_path


class _NpyEncFileIO(INumpyFileIOCore):
    @classmethod
    def load(cls, path: pathlib.Path, decrypt_key: bytes = None) -> Any:  # noqa: ANN401
        if decrypt_key is None:
            raise ValueError(
                "decrypt key is None. Cannot decrypt encrypted file."
            )

        return np.load(utils.decrypt_file(decrypt_key, path))

    @classmethod
    def get_save_path(
        cls, output_directory: pathlib.Path, file_basename: str
    ) -> pathlib.Path:
        return output_directory / (
            file_basename + PhlowerFileExtType.NPYENC.value
        )

    @classmethod
    def save(
        cls,
        save_path: pathlib.Path,
        data: ArrayDataType,
        dtype: np.dtype = np.float32,
        encrypt_key: bytes = None,
    ) -> None:
        if encrypt_key is None:
            raise ValueError(
                "Encrption key is None. Cannot create encrypted file."
            )

        bytesio = io.BytesIO()
        np.save(bytesio, data.astype(dtype))
        utils.encrypt_file(encrypt_key, save_path, bytesio)
        return


class _NpzFileIO(INumpyFileIOCore):
    @classmethod
    def load(cls, path: pathlib.Path, decrypt_key: bytes = None) -> Any:  # noqa: ANN401
        return sp.load_npz(path)

    @classmethod
    def get_save_path(
        cls, output_directory: pathlib.Path, file_basename: str
    ) -> pathlib.Path:
        return output_directory / (file_basename + PhlowerFileExtType.NPZ.value)

    @classmethod
    def save(
        cls,
        save_path: pathlib.Path,
        data: ArrayDataType,
        dtype: np.dtype = np.float32,
        encrypt_key: bytes = None,
    ) -> None:
        sp.save_npz(save_path, data.tocoo().astype(dtype))
        return save_path


class _NpzEncFileIO(INumpyFileIOCore):
    @classmethod
    def load(cls, path: pathlib.Path, decrypt_key: bytes = None) -> Any:  # noqa: ANN401:
        if decrypt_key is None:
            raise ValueError("Key is None. Cannot decrypt encrypted file.")

        return sp.load_npz(utils.decrypt_file(decrypt_key, path))

    @classmethod
    def get_save_path(
        cls, output_directory: pathlib.Path, file_basename: str
    ) -> pathlib.Path:
        return output_directory / (
            file_basename + PhlowerFileExtType.NPZENC.value
        )

    @classmethod
    def save(
        cls,
        save_path: pathlib.Path,
        data: ArrayDataType,
        dtype: np.dtype = np.float32,
        encrypt_key: bytes = None,
    ) -> None:
        if encrypt_key is None:
            raise ValueError(
                "Encrption key is None. Cannot create encrypted file."
            )
        bytesio = io.BytesIO()
        sp.save_npz(bytesio, data.tocoo().astype(dtype))
        utils.encrypt_file(encrypt_key, save_path, bytesio)
        return save_path
