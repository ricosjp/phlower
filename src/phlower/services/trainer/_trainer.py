import pathlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self

from phlower._base import PhlowerTensor
from phlower.data import DataLoaderBuilder, LazyPhlowerDataset, LumpedTensorData
from phlower.io import (
    PhlowerCheckpointFile,
    PhlowerYamlFile,
    select_snapshot_file,
)
from phlower.nn import PhlowerGroupModule
from phlower.services.loss_operations import LossCalculator
from phlower.services.trainer._optimizer import PhlowerOptimizerWrapper
from phlower.services.trainer._trainer_logger import LogRecord, LogRecordIO
from phlower.settings import (
    PhlowerModelSetting,
    PhlowerSetting,
    PhlowerTrainerSetting,
)
from phlower.utils import PhlowerProgressBar, StopWatch, get_logger
from phlower.utils.enums import ModelSelectionType
from phlower.utils.exceptions import PhlowerRestartTrainingCompletedError

_logger = get_logger(__name__)


class PhlowerTrainer:
    # @classmethod
    # def from_directory(cls, saved_directory: pathlib.Path) -> Self:
    #     ph_directory = PhlowerDirectory(saved_directory)
    #     yaml_file = ph_directory.find_yaml_file(cls._SAVED_SETTING_NAME)
    #     setting = PhlowerSetting.read_yaml(yaml_file)
    #     return cls.from_setting(setting)

    @classmethod
    def from_setting(cls, setting: PhlowerSetting) -> Self:
        if (setting.model is None) or (setting.training is None):
            raise ValueError("setting content about scaling is not found.")

        setting.model.resolve()
        return cls(setting.model, setting.training)

    def __init__(
        self,
        model_setting: PhlowerModelSetting,
        trainer_setting: PhlowerTrainerSetting,
    ):
        # NOTE: Must Call at first
        self._fix_seed(trainer_setting.random_seed)

        self._model_setting = model_setting
        self._trainer_setting = trainer_setting

        self._progress_bar = PhlowerProgressBar(
            total=self._trainer_setting.n_epoch
        )

        # initialize model
        self._model = PhlowerGroupModule.from_setting(
            self._model_setting.network
        )
        self._model.to(self._trainer_setting.device)
        self._scheduled_optimizer = PhlowerOptimizerWrapper.from_setting(
            self._trainer_setting, model=self._model
        )
        self._start_epoch = 0
        self._offset_time = 0.0

    def _fix_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def train(
        self,
        output_directory: pathlib.Path,
        train_directories: list[pathlib.Path],
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ) -> PhlowerTensor:
        record_io = LogRecordIO(file_path=output_directory / "log.csv")
        validation_directories = validation_directories or []

        if self._start_epoch == 0:
            # start_epoch > 0 means that this training is restarted.
            self._save_setting(output_directory, encrypt_key=encrypt_key)
            record_io.write_header()

        train_dataset = LazyPhlowerDataset(
            input_settings=self._model_setting.inputs,
            label_settings=self._model_setting.labels,
            field_settings=self._model_setting.fields,
            directories=train_directories,
            decrypt_key=decrypt_key,
        )
        validation_dataset = LazyPhlowerDataset(
            input_settings=self._model_setting.inputs,
            label_settings=self._model_setting.labels,
            field_settings=self._model_setting.fields,
            directories=validation_directories,
            decrypt_key=decrypt_key,
        )

        builder = DataLoaderBuilder.from_setting(self._trainer_setting)
        train_loader = builder.create(
            train_dataset,
            disable_dimensions=disable_dimensions,
        )
        if len(validation_dataset) != 0:
            validation_loader = builder.create(
                validation_dataset,
                disable_dimensions=disable_dimensions,
            )

        loss_function = LossCalculator.from_setting(self._trainer_setting)
        loss: PhlowerTensor | None = None

        tqdm.write(record_io.get_header())
        _timer = StopWatch(offset=self._offset_time)
        _timer.start()
        _train_batch_pbar = PhlowerProgressBar(total=len(train_dataset))
        _val_batch_pbar = PhlowerProgressBar(total=len(validation_dataset))

        for epoch in range(self._start_epoch, self._trainer_setting.n_epoch):
            self._model.train()
            train_losses: list[float] = []

            for tr_batch in train_loader:
                tr_batch: LumpedTensorData
                self._scheduled_optimizer.zero_grad()

                h = self._model.forward(
                    tr_batch.x_data, field_data=tr_batch.field_data
                )

                losses = loss_function.calculate(
                    h, tr_batch.y_data, batch_info_dict=tr_batch.y_batch_info
                )
                loss = loss_function.aggregate(losses)
                loss.backward()
                self._scheduled_optimizer.step_optimizer()

                _last_loss = loss.detach().to_tensor().float().item()
                _train_batch_pbar.update(
                    trick=self._trainer_setting.batch_size,
                    desc=f"training loss: {_last_loss:.3f}",
                )
                train_losses.append(_last_loss)

            self._scheduled_optimizer.step_scheduler()

            if self._trainer_setting.evaluation_for_training:
                train_loss = self._evaluation(
                    train_loader,
                    loss_function=loss_function,
                    pbar=_train_batch_pbar,
                    pbar_title="batch train loss",
                )
            else:
                train_loss = np.average(train_losses)

            if len(validation_dataset) != 0:
                validation_loss = self._evaluation(
                    validation_loader,
                    loss_function=loss_function,
                    pbar=_val_batch_pbar,
                    pbar_title="batch val loss",
                )
            else:
                validation_loss = None

            elapsed_time = _timer.watch()

            log_record = LogRecord(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                elapsed_time=elapsed_time,
            )
            tqdm.write(record_io.to_str(log_record))

            if validation_loss is not None:
                self._progress_bar.update(
                    trick=1, desc=f"val loss {validation_loss:.3f}"
                )
            else:
                self._progress_bar.update(
                    trick=1, desc=f"train loss {train_loss:.3f}"
                )

            record_io.write(log_record)
            self._save_checkpoint(
                output_directory=output_directory,
                epoch=epoch,
                validation_loss=validation_loss,
                elapsed_time=elapsed_time,
            )
        return loss.detach()

    def _evaluation(
        self,
        data_loader: DataLoader,
        loss_function: LossCalculator,
        pbar: PhlowerProgressBar,
        pbar_title: str,
    ) -> float:
        results: list[float] = []

        self._model.eval()
        for _batch in data_loader:
            with torch.no_grad():
                _batch: LumpedTensorData
                h = self._model.forward(
                    _batch.x_data, field_data=_batch.field_data
                )
                val_losses = loss_function.calculate(
                    h,
                    _batch.y_data,
                    batch_info_dict=_batch.y_batch_info,
                )
                val_loss = loss_function.aggregate(val_losses)
                results.append(val_loss.detach().to_tensor().float().item())
            pbar.update(
                trick=self._trainer_setting.batch_size,
                desc=f"{pbar_title}: {results[-1]}",
            )
        return np.average(results)

    def _save_setting(
        self, output_directory: pathlib.Path, encrypt_key: bytes | None = None
    ) -> None:
        dump_setting = PhlowerSetting(
            training=self._trainer_setting, model=self._model_setting
        )
        PhlowerYamlFile.save(
            output_directory=output_directory,
            file_basename="model",
            data=dump_setting.model_dump(),
            encrypt_key=encrypt_key,
            allow_overwrite=False,
        )

    def _save_checkpoint(
        self,
        output_directory: pathlib.Path,
        epoch: int,
        validation_loss: float,
        elapsed_time: float,
        encrypt_key: bytes | None = None,
    ) -> None:
        data = {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": self._model.state_dict(),
            "scheduled_optimizer": self._scheduled_optimizer.state_dict(),
            "elapsed_time": elapsed_time,
        }
        prefix = PhlowerCheckpointFile.get_fixed_prefix()
        file_basename = f"{prefix}{epoch}"
        PhlowerCheckpointFile.save(
            output_directory=output_directory,
            file_basename=file_basename,
            data=data,
            encrypt_key=encrypt_key,
        )
        return

    def reinit_for_restart(
        self,
        restart_directory: pathlib.Path,
        device: str | None = None,
        decrypt_key: bytes | None = None,
    ) -> None:
        # Restore model and optimizer and tqdm
        snapshot_file = select_snapshot_file(
            restart_directory, selection_mode=ModelSelectionType.LATEST.value
        )
        checkpoint = self.load_state(
            snapshot_file=snapshot_file, device=device, decrypt_key=decrypt_key
        )

        self._start_epoch = int(checkpoint["epoch"]) + 1
        self._offset_time = checkpoint["elapsed_time"]

        if self._trainer_setting.n_epoch == self._start_epoch:
            raise PhlowerRestartTrainingCompletedError(
                "Checkpoint at last epoch exists. "
                "Model to restart has already finished"
            )

    def load_state(
        self,
        snapshot_file: PhlowerCheckpointFile,
        device: str | None = None,
        decrypt_key: bytes | None = None,
    ) -> dict:
        # Restore model and optimizer and tqdm
        checkpoint = snapshot_file.load(device=device, decrypt_key=decrypt_key)

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._scheduled_optimizer.load_state_dict(
            checkpoint["scheduled_optimizer"]
        )

        # self.loss = checkpoint['loss']
        _logger.info(
            f"{snapshot_file.file_path} is successfully " "loaded for restart."
        )
        return checkpoint
