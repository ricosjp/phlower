from __future__ import annotations

import json
import pathlib

import pydantic
from packaging.version import Version
from pydantic import dataclasses as dc
from pydantic_core import ErrorDetails
from typing_extensions import Self

from phlower.io import PhlowerYamlFile
from phlower.settings._model_setting import PhlowerModelSetting
from phlower.settings._scaling_setting import PhlowerScalingSetting
from phlower.settings._trainer_setting import PhlowerTrainerSetting
from phlower.utils import get_logger
from phlower.utils.enums import ModelSelectionType
from phlower.version import __version__

_logger = get_logger(__name__)


class PhlowerSetting(pydantic.BaseModel):
    version: str = __version__

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
    def read_yaml(
        cls,
        file_path: pathlib.Path | str | PhlowerYamlFile,
        decrypt_key: bytes | None = None,
    ) -> Self:
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
        try:
            return PhlowerSetting(**data)
        except pydantic.ValidationError as ex:
            raise ValueError(
                "Invalid contents are found in the input setting file. "
                "Details are shown below. \n"
                f"{_format_errors(ex.errors())}"
            ) from ex


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class PhlowerPredictorSetting:
    selection_mode: str
    """
    Define method to select checkpoint file.
    Choose from "best", "latest", "train_best", "specified"
    """

    device: str = "cpu"
    """
    device name. Defaults to cpu
    """

    log_file_name: str = "log"
    """
    name of log file. Defaults to "log"
    """

    saved_setting_filename: str = "model"
    """
    file name of pretrained model setting. Defaults to "model"
    """

    batch_size: int = 1
    """
    batch size. Defaults to 1
    """

    num_workers: int = 1
    """
    the number of cores. Defaults to 1.
    """

    non_blocking: bool = False

    random_seed: int = 0
    """
    random seed. Defaults to 0
    """

    target_epoch: int = -1
    """
    target_epoch specifies the number of snapshot. Defaults to -1.
    """

    @pydantic.field_validator("selection_mode")
    @classmethod
    def check_valid_selection_mode(cls, name: str) -> str:
        names = [v.value for v in ModelSelectionType]
        if name not in names:
            raise ValueError(f"{name} selection mode does not exist.")
        return name

    @pydantic.field_validator("target_epoch")
    @classmethod
    def check_valid_target_epoch(cls, val: int) -> int:
        if cls.selection_mode != "specified":
            if val >= 0:
                raise ValueError(
                    "target_epoch should be None or negative"
                    f"if selection_mode is not specified but {val}"
                )
        else:
            if val < 0:
                raise ValueError(
                    f"target_epoch should be non negative value "
                    f"but {val}"
                )

        return val


def _format_errors(errors: list[ErrorDetails]) -> str:
    data = {"errors": []}

    for error in errors:
        _data = {
            "type": error["type"],
            "location": error["loc"],
            "message": error["msg"],
            "your input": error["input"],
        }
        data["errors"].append(_data)

    return json.dumps(data, indent=4)
