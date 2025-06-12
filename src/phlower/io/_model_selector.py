import abc
import os
import pathlib
from typing import Literal

import numpy as np
import pandas as pd

from phlower.io._directory import PhlowerDirectory
from phlower.io._files import IPhlowerCheckpointFile
from phlower.utils import get_logger
from phlower.utils.enums import ModelSelectionType

_logger = get_logger(__name__)


def select_snapshot_file(
    directory: os.PathLike | PhlowerDirectory,
    selection_mode: Literal["best", "latest", "train_best", "specified"],
    target_epoch: int | None = None,
    **kwards,
) -> IPhlowerCheckpointFile:
    selector = ModelSelectorBuilder.create(selection_mode)

    ph_dir = PhlowerDirectory(directory)
    snapshots = ph_dir.find_snapshot_files()

    if len(snapshots) == 0:
        raise FileNotFoundError(f"snapshot file does not exist in {directory}")

    log_file_path = ph_dir.find_csv_file("log", allow_missing=True)
    file_path = selector.select_model(
        snapshots=snapshots, log_file=log_file_path, target_epoch=target_epoch
    )

    _logger.info(f"{file_path.file_path} is selected.")
    return file_path


class IModelSelector(metaclass=abc.ABCMeta):
    @staticmethod
    @abc.abstractmethod
    def select_model(
        snapshots: list[IPhlowerCheckpointFile],
        log_file: pathlib.Path | None = None,
        **kwards,
    ) -> IPhlowerCheckpointFile:
        raise NotImplementedError()


class ModelSelectorBuilder:
    @staticmethod
    def create(selection_name: str) -> IModelSelector:
        _selector_dict = {
            ModelSelectionType.BEST.value: BestModelSelector,
            ModelSelectionType.LATEST.value: LatestModelSelector,
            ModelSelectionType.SPECIFIED.value: SpecifiedModelSelector,
            ModelSelectionType.TRAIN_BEST.value: TrainBestModelSelector,
        }
        if selection_name not in _selector_dict.keys():
            raise NotImplementedError(
                f"Selection Mode: {selection_name} is not implemented."
            )
        return _selector_dict[selection_name]


class BestModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        snapshots: list[IPhlowerCheckpointFile],
        log_file: pathlib.Path | None = None,
        **kwards,
    ) -> IPhlowerCheckpointFile:
        if log_file is None or (not log_file.exists()):
            raise ValueError(f"log file is missing. {log_file}")

        df = pd.read_csv(
            log_file, header=0, index_col=None, skipinitialspace=True
        )
        if np.any(np.isnan(df["validation_loss"].to_numpy())):
            raise ValueError("NaN value is found in validation result.")

        best_epoch = df["epoch"].iloc[df["validation_loss"].idxmin()]

        target_snapshot = [p for p in snapshots if p.epoch == best_epoch][0]
        return target_snapshot


class LatestModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        snapshots: list[IPhlowerCheckpointFile],
        log_file: pathlib.Path | None = None,
        **kwards,
    ) -> IPhlowerCheckpointFile:
        max_epoch = max([p.epoch for p in snapshots])
        target_snapshot = [p for p in snapshots if p.epoch == max_epoch][0]
        return target_snapshot


class TrainBestModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        snapshots: list[IPhlowerCheckpointFile],
        log_file: pathlib.Path | None = None,
        **kwards,
    ) -> IPhlowerCheckpointFile:
        df = pd.read_csv(
            log_file, header=0, index_col=None, skipinitialspace=True
        )
        best_epoch = df["epoch"].iloc[df["train_loss"].idxmin()]

        target_snapshot: IPhlowerCheckpointFile = [
            p for p in snapshots if p.epoch == best_epoch
        ][0]
        return target_snapshot


class SpecifiedModelSelector(IModelSelector):
    @staticmethod
    def select_model(
        snapshots: list[IPhlowerCheckpointFile],
        log_file: pathlib.Path | None = None,
        *,
        target_epoch: int | None = None,
        **kwards,
    ) -> IPhlowerCheckpointFile:
        if target_epoch is None:
            raise ValueError(
                "Specify target_epoch when using specified selection mode."
            )
        if target_epoch < 0:
            raise ValueError(
                f"Specified target_epoch must be non-negative "
                f"but {target_epoch}."
            )

        target_snapshots: IPhlowerCheckpointFile = [
            p for p in snapshots if p.epoch == target_epoch
        ]

        if len(target_snapshots) == 0:
            raise FileNotFoundError(
                f"File at {target_epoch} epoch does not exist."
            )

        if len(target_snapshots) != 1:
            raise ValueError(
                f"several snapshot files in {target_epoch} are found."
            )

        return target_snapshots[0]
