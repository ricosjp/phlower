from __future__ import annotations

import os
import pathlib
from datetime import timedelta
from typing import Literal, overload

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    fully_shard,
    register_fsdp_forward_method,
)
from torch.nn.parallel import DistributedDataParallel

from phlower.io import (
    PhlowerCheckpointFile,
    PhlowerDirectory,
    PhlowerYamlFile,
    select_snapshot_file,
)
from phlower.nn import PhlowerGroupModule
from phlower.services.loss_operations import LossCalculator
from phlower.services.trainer._dataloader_helper import (
    prepare_dataloader,
    prepare_datasets,
)
from phlower.services.trainer._handlers import PhlowerHandlersRunner
from phlower.services.trainer._loggings import LoggingRunner
from phlower.services.trainer._optimizer import PhlowerOptimizerWrapper
from phlower.services.trainer._runners import EvaluationRunner, TrainingRunner
from phlower.settings import (
    PhlowerDataSetting,
    PhlowerSetting,
    PhlowerTrainerSetting,
)
from phlower.utils import (
    PhlowerProgressBar,
    StopWatch,
    determine_max_process,
    fix_seed,
    get_logger,
)
from phlower.utils.enums import (
    ModelSelectionType,
    PhlowerHandlerTrigger,
    TrainerInitializeType,
    TrainerSavedKeyType,
)
from phlower.utils.exceptions import PhlowerRestartTrainingCompletedError

_logger = get_logger(__name__)


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
        cls,
        model_directory: pathlib.Path,
        decrypt_key: bytes | None = None,
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

        # initialize model
        setting.model.resolve()
        trainer = PhlowerTrainer(setting, restart_directory=model_directory)

        # reinit for restart
        trainer._reinit_for_restart(model_directory, decrypt_key=decrypt_key)
        return trainer

    @classmethod
    def from_setting(
        cls,
        setting: PhlowerSetting,
        decrypt_key: bytes | None = None,
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

        return trainer

    def get_registered_trainer_setting(self) -> PhlowerTrainerSetting:
        """Get registered trainer setting

        Returns:
            PhlowerTrainerSetting: Trainer setting
        """
        return self._setting.training

    def __init__(
        self,
        setting: PhlowerSetting,
        restart_directory: pathlib.Path | None = None,
    ):
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
        fix_seed(self._setting.training.random_seed)

        # check environment
        _check_environment(self._setting)

        # initializer
        self._state_setup = _TrainingStateSetup(
            setting=self._setting,
            restart_directory=restart_directory,
        )

        # initialize handler
        self._handlers = PhlowerHandlersRunner.from_setting(
            self._setting.training
        )

        # initialize loss calculator
        self._loss_calculator = LossCalculator.from_setting(
            self._setting.training
        )
        self._evaluation_runner = EvaluationRunner(
            trainer_setting=self._setting.training,
            loss_calculator=self._loss_calculator,
            handlers=self._handlers,
        )
        self._training_runner = TrainingRunner(
            trainer_setting=self._setting.training,
            loss_calculator=self._loss_calculator,
            handlers=self._handlers,
        )

        # Internal state
        self._start_epoch = 0
        self._offset_time = 0.0

        # NOTE: If this consumes too much memory,
        # we may need to wrap this method with `with init_empty_weights():`
        # and use FSDP2
        self._model = PhlowerGroupModule.from_setting(setting.model.network)

        self._tcp_port: int | None = None
        if self._setting.training.parallel_setting.is_active:
            # NOTE: Ensure tcp_port is determined before calling mp.spawn
            self._tcp_port = self._setting.training.parallel_setting.tcp_port

    def attach_handler(
        self,
        name: str,
        handler: PhlowerHandlersRunner,
        allow_overwrite: bool = False,
    ) -> None:
        """Attach handler to the trainer
        Args:
            name (str): Name of the handler
            handler (PhlowerHandlersRunner): Handler to attach
        Raises:
            ValueError: If handler with the same name already exists
        """
        self._handlers.attach(name, handler, allow_overwrite=allow_overwrite)

    def get_n_handlers(self) -> int:
        """Get the number of handlers

        Returns:
            int: Number of handlers
        """
        return self._handlers.n_handlers

    def _initialize_model(self) -> PhlowerGroupModule:
        _model = PhlowerGroupModule.from_setting(self._setting.model.network)
        if not self._setting.training.parallel_setting.is_active:
            _model.to(self._setting.training.device)
        return _model

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
            loss_keys=self._setting.training.loss_setting.loss_variable_names(),
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

        train_dataset, validation_dataset = prepare_datasets(
            setting=self._setting,
            train_directories=train_directories,
            validation_directories=validation_directories,
            decrypt_key=decrypt_key,
        )
        train_loader, validation_loader = prepare_dataloader(
            setting=self._setting,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            disable_dimensions=disable_dimensions,
        )

        train_last_loss: float | None = None

        _train_batch_pbar = PhlowerProgressBar(total=len(train_directories))
        _val_batch_pbar = PhlowerProgressBar(total=len(validation_directories))

        logging_runner.show_overview(device=self._setting.training.device)

        self._model = self._state_setup.setup_model(
            model=self._model,
            device=self._setting.training.device,
            map_location=self._setting.training.device,
            decrypt_key=decrypt_key,
        )
        _scheduled_optimizer = self._state_setup.setup_scheduled_optimizer(
            model=self._model,
            decrypt_key=decrypt_key,
        )

        _timer = StopWatch(offset=self._offset_time)
        _timer.start()
        for epoch in range(self._start_epoch, self._setting.training.n_epoch):
            self._model.train()

            info = self._training_runner.run(
                epoch,
                output_directory=output_directory,
                model=self._model,
                train_loader=train_loader,
                scheduled_optimizer=_scheduled_optimizer,
                train_pbar=_train_batch_pbar,
            )

            output = self._evaluation_runner.run(
                rank=0,
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
                scheduled_optimizer=_scheduled_optimizer,
                handlers=self._handlers,
                encrypt_key=encrypt_key,
            )

            train_last_loss = output.train_eval_loss

            # Call handlers
            self._handlers.run(
                output,
                trigger=PhlowerHandlerTrigger.epoch_completed,
            )
            if self._handlers.terminate_training:
                _logger.info("Training process is killed by handler.")
                break

        return train_last_loss

    def train_ddp(
        self,
        rank: int,
        world_size: int,
        output_directory: pathlib.Path,
        train_directories: list[pathlib.Path] | None = None,
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ) -> float:
        """
        Train the model with Distributed Data Parallel (DDP)

        Parameters
        ----------
        rank : int
            Rank of the current process
        world_size : int
            Total number of processes
        output_directory : pathlib.Path
            Output directory
        train_directories : list[pathlib.Path] | None, optional
            List of directories containing training data.
            If None, directories defined in the setting are used.
            Default is None.
        validation_directories : list[pathlib.Path] | None, optional
            List of directories containing validation data.
            If None, directories defined in the setting are used.
            Default is None.
        disable_dimensions : bool, optional
            Disable dimensions. Default is False.
        decrypt_key : bytes | None, optional
            Key used for decrypting data files, if necessary. Default is None.
        encrypt_key : bytes | None, optional
            Key used for encrypting output files, if necessary. Default is None.

        Examples
        --------
        >>> import torch.multiprocessing as mp
        >>> trainer = PhlowerTrainer.from_setting(setting)
        >>> mp.spawn(
        ...     trainer.train_ddp,
        ...     args=(world_size, output_directory),
        ...     nprocs=world_size,
        ...     join=True
        ... )
        """

        # setup for parallel training if necessary
        assert self._tcp_port is not None, "tcp_port is not set."
        _setup_parallel_env(
            rank,
            world_size,
            tcp_port=self._tcp_port,
            backend=self._setting.training.parallel_setting.backend,
        )

        data_setting = self._setting.data.upgrade(
            training=train_directories, validation=validation_directories
        )
        if data_setting.is_empty():
            raise ValueError(
                "No training or validation directories are found. "
                "Please check the setting file."
            )

        if rank == 0:
            logging_runner = LoggingRunner(
                output_directory,
                log_every_n_epoch=self._setting.training.log_every_n_epoch,
                loss_keys=self._setting.training.loss_setting.loss_variable_names(),
                parallel_mode=self._setting.training.parallel_setting.parallel_type,
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

        train_dataset, validation_dataset = prepare_datasets(
            setting=self._setting,
            train_directories=train_directories,
            validation_directories=validation_directories,
            decrypt_key=decrypt_key,
        )
        train_loader, validation_loader = prepare_dataloader(
            setting=self._setting,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            disable_dimensions=disable_dimensions,
            run_distributed=True,
        )

        train_last_loss: float | None = None

        _train_batch_pbar = PhlowerProgressBar(total=len(train_directories))
        _val_batch_pbar = PhlowerProgressBar(total=len(validation_directories))

        self._model = self._state_setup.setup_parallel_model(
            model=self._model, rank=rank, decrypt_key=decrypt_key
        )
        _scheduled_optimizer = self._state_setup.setup_scheduled_optimizer(
            model=self._model,
            rank=rank,
            decrypt_key=decrypt_key,
        )

        if rank == 0:
            _timer = StopWatch(offset=self._offset_time)
            _timer.start()
            logging_runner.show_overview(device=self._setting.training.device)
        else:
            _timer = None

        for epoch in range(self._start_epoch, self._setting.training.n_epoch):
            self._model.train()

            info = self._training_runner.parallel_run(
                rank,
                epoch,
                output_directory=output_directory,
                model=self._model,
                train_loader=train_loader,
                scheduled_optimizer=_scheduled_optimizer,
                train_pbar=_train_batch_pbar,
            )

            output = self._evaluation_runner.parallel_run(
                rank=rank,
                info=info,
                model=self._model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                train_pbar=_train_batch_pbar,
                validation_pbar=_val_batch_pbar,
                timer=_timer,
            )

            # Valid output object is returned only from rank 0
            if rank == 0:
                logging_runner.run(
                    output=output,
                    model=self._model,
                    scheduled_optimizer=_scheduled_optimizer,
                    handlers=self._handlers,
                    encrypt_key=encrypt_key,
                )

                train_last_loss = output.train_eval_loss

                # Call handlers
                self._handlers.run(
                    output,
                    trigger=PhlowerHandlerTrigger.epoch_completed,
                )
                if self._handlers.terminate_training:
                    _logger.info("Training process is killed by handler.")
                    break

        _cleanup_parallel()
        return train_last_loss

    def train_fsdp(
        self,
        rank: int,
        world_size: int,
        output_directory: pathlib.Path,
        train_directories: list[pathlib.Path] | None = None,
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ) -> float:
        """
        Train the model with Fully Shard Data Parallel (FSDP)

        Parameters
        ----------
        rank : int
            Rank of the current process
        world_size : int
            Total number of processes
        output_directory : pathlib.Path
            Output directory
        train_directories : list[pathlib.Path] | None, optional
            List of directories containing training data.
            If None, directories defined in the setting are used.
            Default is None.
        validation_directories : list[pathlib.Path] | None, optional
            List of directories containing validation data.
            If None, directories defined in the setting are used.
            Default is None.
        disable_dimensions : bool, optional
            Disable dimensions. Default is False.
        decrypt_key : bytes | None, optional
            Key used for decrypting data files, if necessary. Default is None.
        encrypt_key : bytes | None, optional
            Key used for encrypting output files, if necessary. Default is None.

        Examples
        --------
        >>> import torch.multiprocessing as mp
        >>> trainer = PhlowerTrainer.from_setting(setting)
        >>> mp.spawn(
        ...     trainer.train_fsdp,
        ...     args=(world_size, output_directory),
        ...     nprocs=world_size,
        ...     join=True
        ... )
        """

        # NOTE: This function is almost same as train_ddp.
        # To avoid unintended mistake, we keep two functions separately so far.
        # In future, we may refactor these functions.

        assert self._setting.training.parallel_setting.parallel_type == "FSDP2"

        # setup for parallel training if necessary
        assert self._tcp_port is not None, "tcp_port is not set."
        _setup_parallel_env(
            rank,
            world_size,
            tcp_port=self._tcp_port,
            device=self._setting.training.get_device(rank),
            backend=self._setting.training.parallel_setting.backend,
        )

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
            loss_keys=self._setting.training.loss_setting.loss_variable_names(),
            parallel_mode=self._setting.training.parallel_setting.parallel_type,
        )

        if rank == 0:
            # when restart training, skip is allowed
            self._save_setting_if_necessary(
                output_directory,
                encrypt_key=encrypt_key,
                data_setting=data_setting,
                skip_if_exist=(self._start_epoch > 0),
            )

        train_directories = data_setting.training
        validation_directories = data_setting.validation

        train_dataset, validation_dataset = prepare_datasets(
            setting=self._setting,
            train_directories=train_directories,
            validation_directories=validation_directories,
            decrypt_key=decrypt_key,
        )
        train_loader, validation_loader = prepare_dataloader(
            setting=self._setting,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            disable_dimensions=disable_dimensions,
            run_distributed=True,
        )

        train_last_loss: float | None = None

        _train_batch_pbar = PhlowerProgressBar(total=len(train_directories))
        _val_batch_pbar = PhlowerProgressBar(total=len(validation_directories))

        self._model = self._state_setup.setup_parallel_model(
            model=self._model, rank=rank, decrypt_key=decrypt_key
        )
        _scheduled_optimizer = self._state_setup.setup_scheduled_optimizer(
            model=self._model,
            rank=rank,
            decrypt_key=decrypt_key,
        )

        if rank == 0:
            _timer = StopWatch(offset=self._offset_time)
            _timer.start()
            logging_runner.show_overview(device=self._setting.training.device)
        else:
            _timer = None

        register_fsdp_forward_method(self._model, "forward")
        for v in self._model._phlower_modules:
            register_fsdp_forward_method(v, "forward")

        for epoch in range(self._start_epoch, self._setting.training.n_epoch):
            info = self._training_runner.parallel_run(
                rank,
                epoch,
                output_directory=output_directory,
                model=self._model,
                train_loader=train_loader,
                scheduled_optimizer=_scheduled_optimizer,
                train_pbar=_train_batch_pbar,
            )

            output = self._evaluation_runner.parallel_run(
                rank=rank,
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
                scheduled_optimizer=_scheduled_optimizer,
                handlers=self._handlers,
                encrypt_key=encrypt_key,
                fsdp_sharded=True,
                rank=rank,
            )

            train_last_loss = output.train_eval_loss

            if rank == 0:
                # Call handlers
                # HACK: NEED TO FIX
                # Currently, some handlers may not support FSDP properly.
                self._handlers.run(
                    output,
                    trigger=PhlowerHandlerTrigger.epoch_completed,
                )
                if self._handlers.terminate_training:
                    _logger.info("Training process is killed by handler.")
                    break

        _cleanup_parallel()
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
        if isinstance(self._model, DistributedDataParallel):
            self._model.module.draw(output_directory=output_directory)
        else:
            self._model.draw(output_directory=output_directory)

    def _reinit_for_restart(
        self,
        restart_directory: pathlib.Path,
        decrypt_key: bytes | None = None,
    ) -> None:
        snapshot_file = select_snapshot_file(
            restart_directory, selection_mode=ModelSelectionType.LATEST.value
        )
        self._load_internal_state(
            snapshot_file=snapshot_file, decrypt_key=decrypt_key
        )

    def load_pretrained(
        self,
        model_directory: pathlib.Path,
        selection_mode: Literal["best", "latest", "train_best", "specified"],
        target_epoch: int | None = None,
        map_location: str | dict | None = None,
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
            checkpoint_file, map_location=map_location, decrypt_key=decrypt_key
        )

    def _load_internal_state(
        self,
        snapshot_file: PhlowerCheckpointFile,
        device: str | None = None,
        decrypt_key: bytes | None = None,
    ) -> None:
        # Restore model and optimizer and tqdm
        checkpoint = snapshot_file.load(
            map_location=device,
            weights_only=False,
            decrypt_key=decrypt_key,
        )

        self._handlers.load_state_dict(checkpoint["handler_runners"])

        self._start_epoch = int(checkpoint["epoch"]) + 1
        self._offset_time = checkpoint["elapsed_time"]

        if self._setting.training.n_epoch == self._start_epoch:
            raise PhlowerRestartTrainingCompletedError(
                "Checkpoint at last epoch exists. "
                "Model to restart has already finished. "
                f"{self._setting.training.n_epoch} epochs."
            )

        # self.loss = checkpoint['loss']
        _logger.info(
            f"{snapshot_file.file_path} is successfully " "loaded for restart."
        )


def _setup_parallel_env(
    rank: int,
    world_size: int,
    tcp_port: int,
    device: torch.device | None = None,
    backend: str = "nccl",
) -> None:
    # initialize the process group
    if backend == "nccl":
        torch.cuda.set_device(rank)

    # set the address and port for the process group
    # This setting assumes that all processes are
    #  running on the same machine.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(tcp_port)
    os.environ["RANK"] = str(0)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    _logger.info(f"Rank: {rank} initialized")

    dist.init_process_group(
        backend,
        device_id=device,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )


def _cleanup_parallel():
    dist.destroy_process_group()


class _TrainingStateSetup:
    def __init__(
        self,
        setting: PhlowerSetting,
        restart_directory: pathlib.Path | None = None,
    ):
        self._setting = setting
        self._restart_directory = restart_directory
        self._checkpoint_file = self._determine_checkpoint_file()

    def _determine_checkpoint_file(self) -> PhlowerCheckpointFile | None:
        init_setting = self._setting.training.initializer_setting
        match init_setting.type_name:
            case TrainerInitializeType.none:
                return None
            case TrainerInitializeType.pretrained:
                assert init_setting.reference_directory is not None
                return select_snapshot_file(
                    directory=init_setting.reference_directory,
                    selection_mode="best",
                )
            case TrainerInitializeType.restart:
                assert self._restart_directory is not None
                return select_snapshot_file(
                    self._restart_directory, selection_mode="latest"
                )
            case _:
                raise NotImplementedError(
                    f"Initialize way for {init_setting.type_name} "
                    "is not implemented."
                )

    def setup_model(
        self,
        model: PhlowerGroupModule,
        device: str,
        map_location: str | dict | None = None,
        decrypt_key: bytes | None = None,
    ) -> PhlowerGroupModule:
        model.to(device, non_blocking=self._setting.training.non_blocking)

        init_setting = self._setting.training.initializer_setting
        match init_setting.type_name:
            case TrainerInitializeType.none:
                return model
            case TrainerInitializeType.pretrained:
                model.load_checkpoint_file(
                    self._checkpoint_file,
                    map_location=map_location,
                    decrypt_key=decrypt_key,
                )
                return model
            case TrainerInitializeType.restart:
                model.load_checkpoint_file(
                    self._checkpoint_file,
                    map_location=map_location,
                    decrypt_key=decrypt_key,
                )
                return model
            case _:
                raise NotImplementedError(
                    f"Initialize way for {init_setting.type_name} "
                    "is not implemented."
                )

    def setup_scheduled_optimizer(
        self,
        model: PhlowerGroupModule,
        rank: int | None = None,
        decrypt_key: bytes | None = None,
    ) -> PhlowerOptimizerWrapper:
        _scheduled_optimizer = PhlowerOptimizerWrapper.from_setting(
            self._setting.training, model=model
        )
        init_setting = self._setting.training.initializer_setting

        map_location = self._get_map_location(rank)
        match init_setting.type_name:
            case TrainerInitializeType.none:
                return _scheduled_optimizer

            case TrainerInitializeType.pretrained:
                return _scheduled_optimizer

            case TrainerInitializeType.restart:
                assert self._restart_directory is not None
                _checkpoint = self._checkpoint_file.load(
                    map_location=map_location,
                    weights_only=False,
                    decrypt_key=decrypt_key,
                )
                _scheduled_optimizer.load_state_dict(
                    _checkpoint[TrainerSavedKeyType.scheduled_optimizer.value]
                )
                return _scheduled_optimizer

    def setup_parallel_model(
        self,
        model: PhlowerGroupModule,
        rank: int,
        decrypt_key: bytes | None = None,
    ) -> DistributedDataParallel | FullyShardedDataParallel:
        parallel_setting = self._setting.training.parallel_setting
        match parallel_setting.parallel_type:
            case "DDP":
                return self._setup_ddp_model(
                    model=model, rank=rank, decrypt_key=decrypt_key
                )
            case "FSDP2":
                return self._setup_fsdp2_model(
                    model=model, rank=rank, decrypt_key=decrypt_key
                )
            case _:
                raise NotImplementedError(
                    f"Parallel type {parallel_setting.parallel_type} "
                    "is not implemented."
                )

    def _setup_fsdp2_model(
        self,
        model: PhlowerGroupModule,
        rank: int,
        decrypt_key: bytes | None = None,
    ) -> FullyShardedDataParallel:
        parallel_setting = self._setting.training.parallel_setting
        assert parallel_setting.is_active
        assert parallel_setting.parallel_type == "FSDP2"

        world_size = parallel_setting.world_size
        device = f"cuda:{rank}"
        mesh = init_device_mesh("cuda", (world_size,))

        # So far, only reshard_after_forward is supported
        fsdp_kwargs = {
            "mesh": mesh,
            "reshard_after_forward": parallel_setting.reshard_after_forward,
        }

        # Only children modules under top GROUP module are wrapped with FSDP
        for i, module in enumerate(model._phlower_modules):
            model._phlower_modules[i] = fully_shard(
                module,
                **fsdp_kwargs,
            )
        model = fully_shard(model, **fsdp_kwargs)

        init_setting = self._setting.training.initializer_setting
        if init_setting.type_name == TrainerInitializeType.none:
            model.to(device, non_blocking=self._setting.training.non_blocking)
            return model

        if rank == 0:
            model = self.setup_model(
                model=model,
                device=device,
                map_location="cpu",
                decrypt_key=decrypt_key,
            )
            state_dict = model.state_dict()
        else:
            state_dict = None

        set_model_state_dict(
            model,
            model_state_dict=state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )

        return model

    def _setup_ddp_model(
        self,
        model: PhlowerGroupModule,
        rank: int,
        decrypt_key: bytes | None = None,
    ) -> DistributedDataParallel:
        parallel_setting = self._setting.training.parallel_setting
        assert parallel_setting.is_active
        assert parallel_setting.parallel_type == "DDP"

        match parallel_setting.backend:
            case "nccl":
                map_location = self._get_map_location(rank)
                model = self.setup_model(
                    model=model,
                    device=f"cuda:{rank}",
                    map_location=map_location,
                    decrypt_key=decrypt_key,
                )
                return DistributedDataParallel(
                    model,
                    device_ids=[rank],
                )
            case "gloo":
                model = self.setup_model(
                    model=model,
                    device="cpu",
                    decrypt_key=decrypt_key,
                )
                return DistributedDataParallel(model)
            case _:
                raise NotImplementedError(
                    f"Backend {parallel_setting.backend} is not implemented."
                )

    def _get_map_location(
        self,
        rank: int | None = None,
    ) -> str | dict | None:
        if rank is None:
            return self._setting.training.device

        parallel_setting = self._setting.training.parallel_setting
        match parallel_setting.backend:
            case "nccl":
                return {"cuda:0": f"cuda:{rank}"}
            case "gloo":
                return "cpu"
            case _:
                raise NotImplementedError(
                    f"Backend {parallel_setting.backend} is not implemented."
                )


def _check_environment(setting: PhlowerSetting) -> None:
    if setting.training.device == "cpu":
        return

    if torch.cuda.is_available() is False:
        raise ValueError(
            "CUDA is not available. Please check your environment."
        )

    parallel_setting = setting.training.parallel_setting
    if parallel_setting.is_active is False:
        return

    n_gpus = torch.cuda.device_count()
    if n_gpus < parallel_setting.world_size:
        raise ValueError(
            f"n_gpus {parallel_setting.world_size} is larger than "
            f"available gpus {n_gpus}"
        )

    n_processes = determine_max_process()
    if n_processes < parallel_setting.world_size:
        raise ValueError(
            f"n_gpus {parallel_setting.world_size} is larger than "
            f"available processes {n_processes}"
        )
