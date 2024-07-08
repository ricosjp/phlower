import io
import pathlib
from typing import Any

import yaml

from phlower import utils
from phlower.utils.enums import PhlowerFileExtType

from .interface import IPhlowerYamlFile


class PhlowerYamlFile(IPhlowerYamlFile):
    def __init__(self, path: pathlib.Path) -> None:
        ext = self._check_extension_type(path)
        self._path = path
        self._ext_type = ext

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self._path}"

    def _check_extension_type(self, path: pathlib.Path) -> PhlowerFileExtType:
        extensions = [
            PhlowerFileExtType.YAML,
            PhlowerFileExtType.YAMLENC,
            PhlowerFileExtType.YML,
            PhlowerFileExtType.YMLENC,
        ]
        for ext in extensions:
            if path.name.endswith(ext.value):
                return ext

        raise NotImplementedError(f"Unknown file extension: {path}")

    @property
    def is_encrypted(self) -> bool:
        return self._ext_type in [
            PhlowerFileExtType.YAMLENC,
            PhlowerFileExtType.YMLENC,
        ]

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(self, *, decrypt_key: bytes = None) -> dict:
        if self.is_encrypted:
            return self._load_encrypted(decrypt_key=decrypt_key)
        else:
            return self._load()

    def _load(self) -> dict:
        parameters = _load_yaml(self._path)
        return parameters

    def _load_encrypted(self, decrypt_key: bytes) -> dict:
        if decrypt_key is None:
            raise ValueError("Key is None. Cannot decrypt encrypted file.")

        parameters = _load_yaml(
            utils.decrypt_file(decrypt_key, self._path, return_stringio=True)
        )
        return parameters

    def save(
        self,
        dump_data: object,
        overwrite: bool = False,
        encrypt_key: bytes = None,
    ):
        if not overwrite:
            if self.file_path.exists():
                raise FileExistsError(f"{self.file_path} already exists")

        self.file_path.parent.mkdir(exist_ok=True, parents=True)
        if self.is_encrypted:
            self._save_encrypted(dump_data=dump_data, key=encrypt_key)
        else:
            self._save(dump_data)

    def _save_encrypted(self, dump_data: object, key: bytes) -> None:
        if key is None:
            raise ValueError(
                f"key is empty when encrpting file: {self.file_path}"
            )
        data_string: str = yaml.dump(dump_data, None)
        bio = io.BytesIO(data_string.encode("utf-8"))
        utils.encrypt_file(key, self.file_path, bio)

    def _save(self, dump_data: object) -> None:
        with open(self.file_path, "w") as fw:
            yaml.dump(dump_data, fw)


def _load_yaml_file(file_path: str | pathlib.Path):
    """Load YAML file.

    Parameters
    ----------
    file_name: str or pathlib.Path
        YAML file name.

    Returns
    --------
    dict_data: dict
        YAML contents.
    """
    with open(file_path) as f:
        dict_data = yaml.load(f, Loader=yaml.SafeLoader)
    return dict_data


def _load_yaml(source: str | pathlib.Path | io.TextIOBase | Any):
    """Load YAML source.

    Parameters
    ----------
    source: File-like object or str or pathlib.Path

    Returns
    --------
    dict_data: dict
        YAML contents.
    """
    if isinstance(source, io.TextIOBase):
        return yaml.load(source, Loader=yaml.SafeLoader)
    if isinstance(source, str):
        return yaml.load(source, Loader=yaml.SafeLoader)
    if isinstance(source, pathlib.Path):
        return _load_yaml_file(source)

    raise ValueError(f"Input type {source.__class__} not understood")
