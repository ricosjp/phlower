import io
import pathlib

import yaml
from typing_extensions import Self

from phlower import utils
from phlower.io._files._interface import IPhlowerYamlFile, to_pathlib_object
from phlower.utils import get_logger
from phlower.utils.enums import PhlowerFileExtType

_logger = get_logger(__name__)


class PhlowerYamlFile(IPhlowerYamlFile):
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
    ) -> Self:
        ext = (
            PhlowerFileExtType.YML.value
            if encrypt_key is None
            else PhlowerFileExtType.YMLENC.value
        )
        file_path = output_directory / f"{file_basename}{ext}"

        if not allow_overwrite:
            if file_path.exists():
                raise FileExistsError(f"{file_path} already exists.")

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if encrypt_key is None:
            _save(file_path, data)
        else:
            _save_encrypted(file_path, data, key=encrypt_key)
        _logger.info(f"{file_basename} is saved in: {file_path}")
        return cls(file_path)

    def __init__(self, path: pathlib.Path | str | IPhlowerYamlFile) -> None:
        path = to_pathlib_object(path)
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
    def name(self) -> str:
        return self.file_path.name

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


def _save_encrypted(
    file_path: pathlib.Path, dump_data: object, key: bytes
) -> None:
    if key is None:
        raise ValueError(f"key is empty when encrpting file: {file_path}")
    data_string: str = yaml.dump(dump_data, None)
    bio = io.BytesIO(data_string.encode("utf-8"))
    utils.encrypt_file(key, file_path, bio)


def _save(file_path: pathlib.Path, dump_data: object) -> None:
    with open(file_path, "w") as fw:
        yaml.dump(dump_data, fw)


def _load_yaml_file(file_path: str | pathlib.Path) -> dict:
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


def _load_yaml(source: str | pathlib.Path | io.TextIOBase) -> dict:
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
