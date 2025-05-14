from __future__ import annotations

import pathlib
import random
from typing import Literal, overload

import numpy as np
import torch
from torch.utils.data import DataLoader

from phlower._base import PhlowerTensor
from phlower.data import DataLoaderBuilder, LazyPhlowerDataset, LumpedTensorData
from phlower.io import (
    PhlowerCheckpointFile,
    PhlowerDirectory,
    PhlowerYamlFile,
    select_snapshot_file,
)
from phlower.nn import PhlowerGroupModule
from phlower.services.loss_operations import LossCalculator
from phlower.services.trainer._handlers import PhlowerHandlersRunner
from phlower.services.trainer._loggings import LoggingRunner
from phlower.services.trainer._optimizer import PhlowerOptimizerWrapper
from phlower.settings import (
    PhlowerDataSetting,
    PhlowerSetting,
    PhlowerTrainerSetting,
)
from phlower.utils import PhlowerProgressBar, StopWatch, get_logger
from phlower.utils.enums import (
    ModelSelectionType,
    TrainerInitializeType,
    TrainerSavedKeyType,
)
from phlower.utils.exceptions import PhlowerRestartTrainingCompletedError
from phlower.utils.typing import (
    AfterEpochTrainingInfo,
    AfterEvaluationOutput,
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
            output_directory=info.output_directory,
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
    """
    PhlowerTrainer is a class that manages the training process.

    Examples
    --------
    >>> trainer = PhlowerTrainer.from_setting(setting)
    >>> trainer.train(
    ...     output_directory,
    ...     train_directories,
    ...     validation_directories
    ... )
    """

    _SAVED_SETTING_NAME: str = "model"

    @classmethod
    def restart_from(
        cls, model_directory: pathlib.Path, decrypt_key: bytes | None = None
    ) -> PhlowerTrainer:
        """Restart PhlowerTrainer from model directory

        Args:
            model_directory: pathlib.Path
                Model directory

        Returns:
            PhlowerTrainer: PhlowerTrainer
        """
        ph_directory = PhlowerDirectory(model_directory)
        yaml_file = ph_directory.find_yaml_file(cls._SAVED_SETTING_NAME)
        setting = PhlowerSetting.read_yaml(yaml_file, decrypt_key=decrypt_key)
        trainer = cls.from_setting(setting)
        trainer._reinit_for_restart(model_directory, decrypt_key=decrypt_key)
        return trainer

    @classmethod
    def from_setting(
        cls, setting: PhlowerSetting, decrypt_key: bytes | None = None
    ) -> PhlowerTrainer:
        """Create PhlowerTrainer from PhlowerSetting

        Args:
            setting: PhlowerSetting
                PhlowerSetting

        Returns:
            PhlowerTrainer: PhlowerTrainer
        """
        if (setting.model is None) or (setting.training is None):
            raise ValueError(
                "setting content for training or model is not found."
            )

        init_setting = setting.training.initializer_setting
        if init_setting.type_name == TrainerInitializeType.restart:
            trainer = PhlowerTrainer.restart_from(
                init_setting.reference_directory, decrypt_key=decrypt_key
            )
            return trainer

        setting.model.resolve()
        trainer = PhlowerTrainer(setting)

        if init_setting.type_name == TrainerInitializeType.none:
            return trainer

        if init_setting.type_name == TrainerInitializeType.pretrained:
            trainer.load_pretrained(
                model_directory=init_setting.reference_directory,
                selection_mode="best",
                decrypt_key=decrypt_key,
            )
            return trainer

        raise NotImplementedError(
            f"Initialize way for {init_setting.type_name} is not implemented."
        )

    def get_registered_trainer_setting(self) -> PhlowerTrainerSetting:
        """Get registered trainer setting

        Returns:
            PhlowerTrainerSetting: Trainer setting
        """
        return self._setting.training

    def __init__(self, setting: PhlowerSetting):
        """Initialize PhlowerTrainer without updating trainer's
         inner state.
        If you want to initialize PhlowerTrainer following to
         your setting such as `restart` or `pretrained`,
         please use `from_setting` method.

        Args:
            setting (PhlowerSetting): setting of phlower

        """
        self._setting = setting

        if self._setting.model is None:
            raise ValueError("setting content for model is not found.")
        if self._setting.training is None:
            raise ValueError("setting content for training is not found.")

        # NOTE: Must Call at first
        self._fix_seed(self._setting.training.random_seed)

        # initialize model
        self._model = PhlowerGroupModule.from_setting(
            self._setting.model.network
        )
        self._model.to(self._setting.training.device)
        self._scheduled_optimizer = PhlowerOptimizerWrapper.from_setting(
            self._setting.training, model=self._model
        )

        # initialize loss calculator
        self._loss_calculator = LossCalculator.from_setting(
            self._setting.training
        )
        self._evaluation_runner = _EvaluationRunner(
            trainer_setting=self._setting.training,
            loss_calculator=self._loss_calculator,
        )

        # initialize handler
        self._handlers = PhlowerHandlersRunner.from_setting(
            self._setting.training
        )

        # Internal state
        self._start_epoch = 0
        self._offset_time = 0.0

    def get_n_handlers(self) -> int:
        """Get the number of handlers

        Returns:
            int: Number of handlers
        """
        return self._handlers.n_handlers

    def _fix_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _initialize_model(self) -> PhlowerGroupModule:
        _model = PhlowerGroupModule.from_setting(self._setting.model.network)
        _model.to(self._setting.training.device)

    def _prepare_dataloader(
        self,
        train_directories: list[pathlib.Path],
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
    ) -> tuple[DataLoader, DataLoader | None]:
        train_dataset = LazyPhlowerDataset(
            input_settings=self._setting.model.inputs,
            label_settings=self._setting.model.labels,
            field_settings=self._setting.model.fields,
            directories=train_directories,
            decrypt_key=decrypt_key,
        )
        validation_dataset = LazyPhlowerDataset(
            input_settings=self._setting.model.inputs,
            label_settings=self._setting.model.labels,
            field_settings=self._setting.model.fields,
            directories=validation_directories,
            decrypt_key=decrypt_key,
        )

        builder = DataLoaderBuilder.from_setting(self._setting.training)
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

        self._scheduled_optimizer.step_optimizer()
        self._scheduled_optimizer.zero_grad()
        _last_loss = loss.detach().to_tensor().float().item()

        del loss
        # NOTE: This is necessary to use less memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return _last_loss

    @overload
    def train(
        self,
        output_directory: pathlib.Path,
        disable_dimensions: bool = False,
        random_seed: int | None = None,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ) -> float:
        """
        Train model using directories defined in the setting

        Args:
            output_directory: pathlib.Path
                Output directory
            disable_dimensions: bool
                Disable dimensions. Defaults to False.
            decrypt_key: bytes | None
                Decrypt key. Defaults to None.
            encrypt_key: bytes | None
                Encrypt key. Defaults to None.
        Returns:
            float: Last loss
        """
        ...

    @overload
    def train(
        self,
        output_directory: pathlib.Path,
        train_directories: list[pathlib.Path],
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ) -> float:
        """Train the model with given directories

        Args:
            output_directory: pathlib.Path
                Output directory
            train_directories: list[pathlib.Path]
                Train directories
            validation_directories: list[pathlib.Path] | None
                Validation directories. Defaults to None.
            disable_dimensions: bool
                Disable dimensions. Defaults to False.
            decrypt_key: bytes | None
                Decrypt key. Defaults to None.
            encrypt_key: bytes | None
                Encrypt key. Defaults to None.

        Returns:
            float: Last loss
        """

    def train(
        self,
        output_directory: pathlib.Path,
        train_directories: list[pathlib.Path] | None = None,
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ) -> float:
        data_setting = self._setting.data.upgrade(
            training=train_directories, validation=validation_directories
        )
        if data_setting.is_empty():
            raise ValueError(
                "No training or validation directories are found. "
                "Please check the setting file."
            )

        logging_runner = LoggingRunner(
            output_directory,
            log_every_n_epoch=self._setting.training.log_every_n_epoch,
        )

        # when restart training, skip is allowed
        self._save_setting_if_necessary(
            output_directory,
            encrypt_key=encrypt_key,
            data_setting=data_setting,
            skip_if_exist=(self._start_epoch > 0),
        )

        train_directories = data_setting.training
        validation_directories = data_setting.validation

        train_loader, validation_loader = self._prepare_dataloader(
            train_directories=train_directories,
            validation_directories=validation_directories,
            disable_dimensions=disable_dimensions,
            decrypt_key=decrypt_key,
        )

        train_last_loss: float | None = None

        _timer = StopWatch(offset=self._offset_time)
        _timer.start()
        _train_batch_pbar = PhlowerProgressBar(total=len(train_directories))
        _val_batch_pbar = PhlowerProgressBar(total=len(validation_directories))

        logging_runner.show_overview(device=self._setting.training.device)

        for epoch in range(self._start_epoch, self._setting.training.n_epoch):
            self._model.train()
            train_losses: list[float] = []

            for tr_batch in train_loader:
                train_last_loss = self._training_batch_step(tr_batch)
                _train_batch_pbar.update(
                    trick=self._setting.training.batch_size,
                    desc=f"training loss: {train_last_loss:.3f}",
                )
                train_losses.append(train_last_loss)

            self._scheduled_optimizer.step_scheduler()
            info = AfterEpochTrainingInfo(
                epoch=epoch,
                train_losses=train_losses,
                output_directory=output_directory,
            )

            output = self._evaluation_runner.run(
                info=info,
                model=self._model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                train_pbar=_train_batch_pbar,
                validation_pbar=_val_batch_pbar,
                timer=_timer,
            )

            logging_runner.run(
                output=output,
                model=self._model,
                scheduled_optimizer=self._scheduled_optimizer,
                handlers=self._handlers,
                encrypt_key=encrypt_key,
            )

            # Call handlers
            self._handlers.run(output)
            if self._handlers.terminate_training:
                _logger.info("Training process is killed by handler.")
                break

        return train_last_loss

    def _save_setting_if_necessary(
        self,
        output_directory: pathlib.Path,
        data_setting: PhlowerDataSetting,
        encrypt_key: bytes | None = None,
        skip_if_exist: bool = False,
    ) -> None:
        dumped_yaml = PhlowerDirectory(output_directory).find_yaml_file(
            file_base_name=self._SAVED_SETTING_NAME, allow_missing=True
        )
        if (dumped_yaml is not None) and skip_if_exist:
            _logger.info("model setting file has already existed. Skip saving.")
            return

        dump_setting = self._setting.model_copy(update={"data": data_setting})
        PhlowerYamlFile.save(
            output_directory=output_directory,
            file_basename=self._SAVED_SETTING_NAME,
            data=dump_setting.model_dump(),
            encrypt_key=encrypt_key,
            allow_overwrite=False,
        )

    def draw_model(self, output_directory: pathlib.Path) -> None:
        """Draw model

        Args:
            output_directory: pathlib.Path
                Output directory
        """
        self._model.draw(output_directory=output_directory)

    def _reinit_for_restart(
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
        """Load pretrained model

        Args:
            model_directory: pathlib.Path
                Model directory
            selection_mode: Literal["best", "latest", "train_best", "specified"]
                Selection mode
            target_epoch: int | None
                Target epoch. Defaults to None.
            device: str | None
                Device. Defaults to None.
            decrypt_key: bytes | None
                Decrypt key. Defaults to None.
        """
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

        if self._setting.training.n_epoch == self._start_epoch:
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
            desc=f"{pbar_title}: {results[-1]:.3f}",
        )
    return np.average(results)
