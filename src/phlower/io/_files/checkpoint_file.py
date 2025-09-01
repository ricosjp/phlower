import io
import os
import pathlib
import re
from typing import Any

import torch

from phlower import utils
from phlower.io._files._interface import (
    IPhlowerCheckpointFile,
    to_pathlib_object,
)
from phlower.utils.enums import PhlowerFileExtType


class PhlowerCheckpointFile(IPhlowerCheckpointFile):
    _FIXED_PREFIX = "snapshot_epoch_"

    @classmethod
    def save(
        cls,
        output_directory: pathlib.Path,
        file_basename: str,
        data: object,
        *,
        encrypt_key: bytes = None,
        allow_overwrite: bool = False,
        **kwargs,
    ) -> IPhlowerCheckpointFile:
        ext = (
            PhlowerFileExtType.PTH.value
            if encrypt_key is None
            else PhlowerFileExtType.PTHENC.value
        )

        file_path = output_directory / f"{file_basename}{ext}"
        if not allow_overwrite:
            if file_path.exists():
                raise FileExistsError(f"{file_path} already exists")

        file_path.parent.mkdir(exist_ok=True, parents=True)
        _save(file_path, data, encrypt_key=encrypt_key)
        return PhlowerCheckpointFile(file_path)

    @classmethod
    def get_fixed_prefix(self) -> int:
        return self._FIXED_PREFIX

    def __init__(self, path: os.PathLike | IPhlowerCheckpointFile):
        path = to_pathlib_object(path)
        self._ext_type = self._check_extension_type(path)
        self._path = path

    def _check_extension_type(self, path: pathlib.Path) -> PhlowerFileExtType:
        extensions = [PhlowerFileExtType.PTH, PhlowerFileExtType.PTHENC]
        for ext in extensions:
            if path.name.endswith(ext.value):
                return ext

        raise NotImplementedError(
            f"Unknown file extension. {path}." ".pth or .pth.enc is allowed"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self._path}"

    def _check_format(self):
        if not self._path.name.startswith(self._FIXED_PREFIX):
            raise ValueError(
                f"File name does not start with '{self._FIXED_PREFIX}': "
                f"{self._path.name}"
            )

    @property
    def name(self) -> str:
        return self.file_path.name

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    @property
    def epoch(self) -> int:
        self._check_format()

        num_epoch = re.search(
            rf"{self._FIXED_PREFIX}(\d+)", self._path.name
        ).groups()[0]
        return int(num_epoch)

    @property
    def is_encrypted(self) -> bool:
        return self._ext_type == PhlowerFileExtType.PTHENC

    def load(
        self,
        map_location: str | dict,
        weights_only: bool = False,
        *,
        decrypt_key: bytes = None,
    ) -> Any:  # noqa: ANN401
        if self.is_encrypted:
            return self._load_encrypted(
                map_location=map_location,
                weights_only=weights_only,
                decrypt_key=decrypt_key,
            )
        else:
            return self._load(
                map_location=map_location, weights_only=weights_only
            )

    def _load(self, map_location: str | dict, weights_only: bool) -> Any:  # noqa: ANN401
        return torch.load(
            self._path, weights_only=weights_only, map_location=map_location
        )

    def _load_encrypted(
        self,
        map_location: str | dict,
        weights_only: bool,
        decrypt_key: bytes | None = None,
    ) -> dict:
        if decrypt_key is None:
            raise ValueError("Feed key to load encrypted model")

        checkpoint = torch.load(
            utils.decrypt_file(decrypt_key, self._path),
            weights_only=weights_only,
            map_location=map_location,
        )
        return checkpoint


def _save(
    file_path: pathlib.Path, dump_data: object, encrypt_key: bytes | None = None
) -> None:
    if encrypt_key is None:
        torch.save(dump_data, file_path)
        return

    buffer = io.BytesIO()
    torch.save(dump_data, buffer)
    utils.encrypt_file(encrypt_key, file_path, buffer)
