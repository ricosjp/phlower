from __future__ import annotations

import pathlib

import pydantic
from pydantic import dataclasses as dc
from typing_extensions import Self

from phlower.io import PhlowerYamlFile
from phlower.settings._model_setting import PhlowerModelSetting
from phlower.settings._scaling_setting import PhlowerScalingSetting
from phlower.settings._trainer_setting import PhlowerTrainerSetting
from phlower.utils.enums import ModelSelectionType


class PhlowerSetting(pydantic.BaseModel):
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
        frozen=True, extra="forbid", arbitrary_types_allowed=True
    )

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
        return PhlowerSetting(**data)


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

    @pydantic.field_validator("selection_mode")
    @classmethod
    def check_valid_selection_mode(cls, name: str) -> str:
        names = [v.value for v in ModelSelectionType]
        if name not in names:
            raise ValueError(f"{name} selection mode does not exist.")
        return name
