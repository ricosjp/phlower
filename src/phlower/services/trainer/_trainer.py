import pathlib
import random
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
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
from phlower.services.trainer._handlers import PhlowerHandlersRunner
from phlower.services.trainer._loggings import LoggingRunner
from phlower.services.trainer._optimizer import PhlowerOptimizerWrapper
from phlower.settings import (
    PhlowerModelSetting,
    PhlowerSetting,
    PhlowerTrainerSetting,
)
from phlower.utils import PhlowerProgressBar, StopWatch, get_logger
from phlower.utils.enums import ModelSelectionType, TrainerSavedKeyType
from phlower.utils.exceptions import PhlowerRestartTrainingCompletedError
from phlower.utils.typing import (
    AfterEpochTrainingInfo,
    AfterEvaluationOutput,
    LossFunctionType,
    PhlowerHandlerType,
)

_logger = get_logger(__name__)


class _EvaluationRunner:
    def __init__(
        self,
        trainer_setting: PhlowerTrainerSetting,
        loss_calculator: LossCalculator,
    ):
        self._trainer_setting = trainer_setting
        self._loss_calculator = loss_calculator

    def run(
        self,
        info: AfterEpochTrainingInfo,
        *,
        model: PhlowerGroupModule,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        train_pbar: PhlowerProgressBar,
        validation_pbar: PhlowerProgressBar,
        timer: StopWatch,
    ) -> AfterEvaluationOutput:
        train_eval_loss = self._evaluate_training(
            info, model=model, train_loader=train_loader, train_pbar=train_pbar
        )

        validation_eval_loss = self._evaluate_validation(
            info,
            model=model,
            validation_loader=validation_loader,
            validation_pbar=validation_pbar,
        )

        return AfterEvaluationOutput(
            epoch=info.epoch,
            train_eval_loss=train_eval_loss,
            validation_eval_loss=validation_eval_loss,
            elapsed_time=timer.watch(),
        )

    def _evaluate_training(
        self,
        info: AfterEpochTrainingInfo,
        model: PhlowerGroupModule,
        train_loader: DataLoader,
        train_pbar: PhlowerProgressBar,
    ) -> float:
        if not self._trainer_setting.evaluation_for_training:
            return np.average(info.train_losses)

        return _evaluation(
            model,
            train_loader,
            loss_function=self._loss_calculator,
            pbar=train_pbar,
            pbar_title="batch train loss",
        )

    def _evaluate_validation(
        self,
        info: AfterEpochTrainingInfo,
        model: PhlowerGroupModule,
        validation_loader: DataLoader | None = None,
        validation_pbar: PhlowerProgressBar | None = None,
    ) -> float | None:
        if validation_loader is None:
            return None

        return _evaluation(
            model,
            validation_loader,
            loss_function=self._loss_calculator,
            pbar=validation_pbar,
            pbar_title="batch val loss",
        )


class PhlowerTrainer:
    # @classmethod
    # def from_directory(cls, saved_directory: pathlib.Path) -> Self:
    #     ph_directory = PhlowerDirectory(saved_directory)
    #     yaml_file = ph_directory.find_yaml_file(cls._SAVED_SETTING_NAME)
    #     setting = PhlowerSetting.read_yaml(yaml_file)
    #     return cls.from_setting(setting)

    @classmethod
    def from_setting(
        cls,
        setting: PhlowerSetting,
        user_loss_functions: dict[str, LossFunctionType] | None = None,
        user_handlers: dict[str, PhlowerHandlerType] | None = None,
    ) -> Self:
        if (setting.model is None) or (setting.training is None):
            raise ValueError(
                "setting content for training or model is not found."
            )

        setting.model.resolve()
        return cls(
            setting.model,
            setting.training,
            user_loss_functions=user_loss_functions,
            user_handlers=user_handlers,
        )

    def __init__(
        self,
        model_setting: PhlowerModelSetting,
        trainer_setting: PhlowerTrainerSetting,
        user_loss_functions: dict[str, LossFunctionType] | None = None,
        user_handlers: dict[str, PhlowerHandlerType] | None = None,
    ):
        # NOTE: Must Call at first
        self._fix_seed(trainer_setting.random_seed)

        self._model_setting = model_setting
        self._trainer_setting = trainer_setting

        self._epoch_progress_bar = PhlowerProgressBar(
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

        # initialize loss calculator
        self._loss_calculator = LossCalculator.from_setting(
            self._trainer_setting, user_loss_functions=user_loss_functions
        )
        self._evaluation_runner = _EvaluationRunner(
            trainer_setting=trainer_setting,
            loss_calculator=self._loss_calculator,
        )

        # initialize handler
        self._handlers = PhlowerHandlersRunner.from_setting(
            trainer_setting, user_defined_handlers=user_handlers
        )

        # Internal state
        self._start_epoch = 0
        self._offset_time = 0.0

    def get_n_handlers(self) -> int:
        return self._handlers.n_handlers

    def _fix_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _initialize_model(self) -> PhlowerGroupModule:
        _model = PhlowerGroupModule.from_setting(self._model_setting.network)
        _model.to(self._trainer_setting.device)

    def _prepare_dataloader(
        self,
        train_directories: list[pathlib.Path],
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
    ) -> tuple[DataLoader, DataLoader | None]:
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

        if len(validation_dataset) == 0:
            return train_loader, None

        validation_loader = builder.create(
            validation_dataset,
            disable_dimensions=disable_dimensions,
        )
        return train_loader, validation_loader

    def _training_batch_step(self, tr_batch: LumpedTensorData) -> PhlowerTensor:
        self._scheduled_optimizer.zero_grad()

        h = self._model.forward(tr_batch.x_data, field_data=tr_batch.field_data)

        losses = self._loss_calculator.calculate(
            h, tr_batch.y_data, batch_info_dict=tr_batch.y_batch_info
        )
        loss = self._loss_calculator.aggregate(losses)
        loss.backward()
        return loss

    def train(
        self,
        output_directory: pathlib.Path,
        train_directories: list[pathlib.Path],
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ) -> PhlowerTensor:
        validation_directories = validation_directories or []
        logging_runner = LoggingRunner(
            output_directory,
            log_every_n_epoch=self._trainer_setting.log_every_n_epoch,
        )

        if self._start_epoch == 0:
            # start_epoch > 0 means that this training is restarted.
            self._save_setting(output_directory, encrypt_key=encrypt_key)

        train_loader, validation_loader = self._prepare_dataloader(
            train_directories=train_directories,
            validation_directories=validation_directories,
            disable_dimensions=disable_dimensions,
            decrypt_key=decrypt_key,
        )

        train_last_loss: PhlowerTensor | None = None

        _timer = StopWatch(offset=self._offset_time)
        _timer.start()
        _train_batch_pbar = PhlowerProgressBar(total=len(train_directories))
        _val_batch_pbar = PhlowerProgressBar(total=len(validation_directories))

        for epoch in range(self._start_epoch, self._trainer_setting.n_epoch):
            self._model.train()
            train_losses: list[float] = []

            for tr_batch in train_loader:
                train_last_loss = self._training_batch_step(tr_batch)
                self._scheduled_optimizer.step_optimizer()
                _last_loss = train_last_loss.detach().to_tensor().float().item()
                _train_batch_pbar.update(
                    trick=self._trainer_setting.batch_size,
                    desc=f"training loss: {_last_loss:.3f}",
                )
                train_losses.append(_last_loss)

            self._scheduled_optimizer.step_scheduler()
            info = AfterEpochTrainingInfo(
                epoch=epoch, train_losses=train_losses
            )

            output = self._evaluation_runner.run(
                info=info,
                model=self._model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                train_pbar=_train_batch_pbar,
                validation_pbar=_val_batch_pbar,
                timer=_timer
            )

            logging_runner.run(
                output=output,
                model=self._model,
                scheduled_optimizer=self._scheduled_optimizer,
                handlers=self._handlers,
                encrypt_key=encrypt_key,
            )

            # Call handlers
            self._handlers(output)
            if self._handlers.terminate_training:
                _logger.info("Training process is killed by handler.")
                break

            # update epoch
            if output.validation_eval_loss is not None:
                self._epoch_progress_bar.update(
                    trick=1, desc=f"val loss {output.validation_eval_loss:.3f}"
                )
            else:
                self._epoch_progress_bar.update(
                    trick=1, desc=f"train loss {output.train_eval_loss:.3f}"
                )

        return train_last_loss

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
        self._load_state(
            snapshot_file=snapshot_file, device=device, decrypt_key=decrypt_key
        )

    def load_pretrained(
        self,
        model_directory: pathlib.Path,
        selection_mode: Literal["best", "latest", "train_best", "specified"],
        target_epoch: int | None = None,
        device: str | None = None,
        decrypt_key: bytes | None = None,
    ) -> None:
        checkpoint_file = select_snapshot_file(
            directory=model_directory,
            selection_mode=selection_mode,
            target_epoch=target_epoch,
        )
        self._model.load_checkpoint_file(
            checkpoint_file, device=device, decrypt_key=decrypt_key
        )

    def _load_state(
        self,
        snapshot_file: PhlowerCheckpointFile,
        device: str | None = None,
        decrypt_key: bytes | None = None,
    ) -> None:
        # Restore model and optimizer and tqdm
        checkpoint = snapshot_file.load(device=device, decrypt_key=decrypt_key)

        self._model.load_state_dict(
            checkpoint[TrainerSavedKeyType.model_state_dict.value]
        )
        self._scheduled_optimizer.load_state_dict(
            checkpoint[TrainerSavedKeyType.scheduled_optimizer.value]
        )
        self._handlers.load_state_dict(checkpoint["handler_runners"])

        self._start_epoch = int(checkpoint["epoch"]) + 1
        self._offset_time = checkpoint["elapsed_time"]

        if self._trainer_setting.n_epoch == self._start_epoch:
            raise PhlowerRestartTrainingCompletedError(
                "Checkpoint at last epoch exists. "
                "Model to restart has already finished"
            )

        # self.loss = checkpoint['loss']
        _logger.info(
            f"{snapshot_file.file_path} is successfully " "loaded for restart."
        )


def _evaluation(
    model: PhlowerGroupModule,
    data_loader: DataLoader | None,
    loss_function: LossCalculator,
    pbar: PhlowerProgressBar,
    pbar_title: str,
) -> float:
    results: list[float] = []

    model.eval()
    for _batch in data_loader:
        with torch.no_grad():
            _batch: LumpedTensorData
            h = model.forward(_batch.x_data, field_data=_batch.field_data)
            val_losses = loss_function.calculate(
                h,
                _batch.y_data,
                batch_info_dict=_batch.y_batch_info,
            )
            val_loss = loss_function.aggregate(val_losses)
            results.append(val_loss.detach().to_tensor().float().item())
        pbar.update(
            trick=_batch.n_data,
            desc=f"{pbar_title}: {results[-1]}",
        )
    return np.average(results)
