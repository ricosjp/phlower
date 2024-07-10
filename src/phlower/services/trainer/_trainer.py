import pathlib
import random

import numpy as np
import torch
from tqdm import tqdm
from typing_extensions import Self

from phlower._base import PhlowerTensor
from phlower.data import DataLoaderBuilder, LazyPhlowerDataset, LumpedTensorData
from phlower.io import PhlowerCheckpointFile, PhlowerYamlFile
from phlower.nn import PhlowerGroupModule
from phlower.services.loss_operations import LossCalculator
from phlower.services.trainer._trainer_logger import LogRecord, LogRecordIO
from phlower.settings import (
    PhlowerModelSetting,
    PhlowerSetting,
    PhlowerTrainerSetting,
)
from phlower.utils import PhlowerProgressBar, StopWatch, get_logger

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

        setting.model.network.resolve(is_first=True)
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
        self._optimizer = torch.optim.SGD(
            self._model.parameters(), lr=self._trainer_setting.lr, momentum=0.9
        )
        self._timer = StopWatch()

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
        encrypt_key: bytes | None = None,
    ) -> PhlowerTensor:
        self._save_setting(output_directory, encrypt_key=encrypt_key)
        record_io = LogRecordIO(file_path=output_directory / "log.csv")
        record_io.write_header()

        train_dataset = LazyPhlowerDataset(
            x_variable_names=self._model_setting.network.get_input_keys(),
            y_variable_names=self._model_setting.network.get_output_keys(),
            support_names=self._model_setting.network.support_names,
            directories=train_directories,
        )
        validation_dataset = LazyPhlowerDataset(
            x_variable_names=self._model_setting.network.get_input_keys(),
            y_variable_names=self._model_setting.network.get_output_keys(),
            support_names=self._model_setting.network.support_names,
            directories=validation_directories,
        )

        builder = DataLoaderBuilder.from_setting(self._trainer_setting)
        train_loader = builder.create(
            train_dataset,
            variable_dimensions=self._model_setting.variable_dimensions,
            disable_dimensions=disable_dimensions,
        )
        validation_loader = builder.create(
            validation_dataset,
            variable_dimensions=self._model_setting.variable_dimensions,
            disable_dimensions=disable_dimensions,
        )

        loss_function = LossCalculator.from_setting(self._trainer_setting)

        loss: PhlowerTensor | None = None

        tqdm.write(record_io.get_header())
        self._timer.start()
        for epoch in range(self._trainer_setting.n_epoch):
            train_losses: list[float] = []
            validation_losses: list[float] = []

            self._model.train()
            for tr_batch in train_loader:
                tr_batch: LumpedTensorData
                self._optimizer.zero_grad()

                h = self._model.forward(
                    tr_batch.x_data, supports=tr_batch.sparse_supports
                )

                losses = loss_function.calculate(
                    h, tr_batch.y_data, batch_info_dict=tr_batch.y_batch_info
                )
                loss = loss_function.aggregate(losses)
                train_losses.append(loss.detach().to_tensor().float().item())
                loss.backward()
                self._optimizer.step()

            self._model.eval()
            for val_batch in validation_loader:
                with torch.no_grad():
                    h = self._model.forward(
                        val_batch.x_data, supports=val_batch.sparse_supports
                    )
                    val_losses = loss_function.calculate(
                        h,
                        val_batch.y_data,
                        batch_info_dict=val_batch.y_batch_info,
                    )
                    val_loss = loss_function.aggregate(val_losses)
                    validation_losses.append(
                        val_loss.detach().to_tensor().float().item()
                    )

            train_loss = np.average(train_losses)
            validation_loss = np.average(validation_losses)
            log_record = LogRecord(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                elapsed_time=self._timer.watch(),
            )
            tqdm.write(record_io.to_str(log_record))

            self._progress_bar.update(
                trick=1, desc=f"val loss {validation_loss:.3f}"
            )

            record_io.write(log_record)
            self._save_checkpoint(
                output_directory=output_directory,
                epoch=epoch,
                validation_loss=validation_loss,
            )
        return loss.detach()

    def _load_state(
        self,
        target_path: pathlib.Path,
        device: str | None = None,
        decrypt_key: bytes | None = None,
    ) -> None:
        # Restore model and optimizer and tqdm
        ...

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
        encrypt_key: bytes | None = None,
    ) -> None:
        data = {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
        }
        PhlowerCheckpointFile.save(
            output_directory=output_directory,
            epoch_number=epoch,
            dump_data=data,
            encrypt_key=encrypt_key,
        )
        return
