from __future__ import annotations

import json
import pathlib

import pydantic
from packaging.version import Version
from pydantic_core import ErrorDetails

from phlower.io import PhlowerYamlFile
from phlower.settings._data_setting import PhlowerDataSetting
from phlower.settings._model_setting import PhlowerModelSetting
from phlower.settings._predictor_setting import PhlowerPredictorSetting
from phlower.settings._scaling_setting import PhlowerScalingSetting
from phlower.settings._trainer_setting import PhlowerTrainerSetting
from phlower.utils import get_logger
from phlower.version import __version__

_logger = get_logger(__name__)


class PhlowerSetting(pydantic.BaseModel):
    version: str = __version__
    """
    version of setting file.
    """

    data: PhlowerDataSetting = pydantic.Field(
        default_factory=PhlowerDataSetting
    )
    """
    Data seting. Defaults to None.
    """

    training: PhlowerTrainerSetting | None = None
    """
    training setting. Defaults to None.
    """

    model: PhlowerModelSetting | None = None
    """
    model setting. Defaults to None.
    """

    scaling: PhlowerScalingSetting | None = None
    """
    scaling setting. Defaults to None.
    """

    prediction: PhlowerPredictorSetting | None = None
    """
    prediction setting. Defaults to None.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(
        frozen=True, arbitrary_types_allowed=True
    )

    @pydantic.field_validator("version")
    @classmethod
    def check_version(cls, version: str) -> str:
        if Version(version) > Version(__version__):
            # When version number in yaml file is larger than program.
            _logger.warning(
                "Version number of input setting file is "
                "higher than program version."
                f"File version: {version}, program: {__version__}"
            )
        return version

    @classmethod
    def read_dictionary(
        cls,
        data: dict,
    ) -> PhlowerSetting:
        """Read dictionary and parse to PhlowerSetting object."""

        try:
            return PhlowerSetting(**data)
        except pydantic.ValidationError as ex:
            raise ValueError(
                "Invalid contents are found in the input setting file. "
                "Details are shown below. \n"
                f"{_format_errors(ex.errors(), ref=data)} \n"
                f"{ex.error_count()} errors are "
                f"found."
            ) from ex

    @classmethod
    def read_yaml(
        cls,
        file_path: pathlib.Path | str | PhlowerYamlFile,
        decrypt_key: bytes | None = None,
    ) -> PhlowerSetting:
        """Read yaml file and parse to PhlowerSetting object.

        Args:
            file_path (pathlib.Path | str | PhlowerYamlFile):
                path to yaml file
            decrypt_key (bytes | None, optional):
                key to decrypt file. Defaults to None.

        Returns:
            Self: PhlowerSetting object
        """
        path = PhlowerYamlFile(file_path)
        data = path.load(decrypt_key=decrypt_key)
        return PhlowerSetting.read_dictionary(data)


def _format_loc(location: list[str], ref: dict | None = None) -> str:
    _msg = []
    for loc in location:
        ref = _access(loc, ref)
        if not isinstance(loc, int):
            _msg.append(loc)
        else:
            _msg.append(f"{_to_order(loc)} item (name: {_access('name', ref)})")

    return " -> ".join(_msg)


def _access(
    key: str | int, ref: dict | list | None = None
) -> dict | list | None:
    if ref is None:
        return None

    if key == "MODULE" or key == "GROUP":
        return ref

    try:
        if isinstance(ref, dict):
            return ref.get(key)

        if isinstance(ref, list):
            return ref[key]
    except Exception:
        return None

    # NOTE: this is unreachable code.
    return None


def _to_order(value: int) -> str:
    value += 1
    if value == 1:
        return "1st"

    if value == 2:
        return "2nd"

    if value == 3:
        return "3rd"

    return f"{value}th"


def _format_errors(errors: list[ErrorDetails], ref: dict) -> str:
    data = {"errors": []}

    for error in errors:
        _data = {
            "error_type": error["type"],
            "error_location": _format_loc(error["loc"], ref),
            "message": error["msg"],
            "your input": error["input"],
        }
        data["errors"].append(_data)

    return json.dumps(data, indent=4)
