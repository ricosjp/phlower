from __future__ import annotations

import pathlib
from collections.abc import Callable
from functools import partial
from typing import Final, NamedTuple, overload

import torch.multiprocessing as mp

from phlower.io import PhlowerYamlFile
from phlower.services.trainer._continue_updater import (
    ContinueParameterUpdatorFactory,
)
from phlower.services.trainer._handlers import IPhlowerHandler
from phlower.services.trainer._trainer import (
    PhlowerTrainer,
    error_handler_wrapper,
)
from phlower.settings import PhlowerSetting
from phlower.settings._continue_settings import (
    ContinueSetting,
)
from phlower.utils import get_logger
from phlower.utils.enums import TrainerInitializeType
from phlower.utils.exceptions import PhlowerRestartTrainingCompletedError

_logger = get_logger(__name__)


class ContinueState(NamedTuple):
    current_count: int
    output_directory: pathlib.Path | None
    setting: PhlowerSetting
    preset_trainer: PhlowerTrainer | None = None

    def initialize_output_directory(
        self, output_directory: pathlib.Path
    ) -> ContinueState:
        return ContinueState(
            current_count=self.current_count,
            output_directory=output_directory,
            setting=self.setting,
            preset_trainer=self.preset_trainer,
        )

    def setup_trainer(self, decrypt_key: bytes | None = None) -> PhlowerTrainer:
        if self.preset_trainer is not None:
            return self.preset_trainer

        # NOTE: setup trainer sometimes takes time to load pretrained model,
        # so we do not setup trainer in __init__.
        return PhlowerTrainer.from_setting(
            setting=self.setting, decrypt_key=decrypt_key
        )


class StateUpdator:
    def __init__(
        self,
        cont0_output_directory: pathlib.Path,
        continue_setting: ContinueSetting,
    ):
        self._cont0_output_directory: Final[pathlib.Path] = (
            cont0_output_directory
        )
        self._param_updator = ContinueParameterUpdatorFactory.create(
            continue_setting
        )

    def update(
        self,
        prev_state: ContinueState,
    ) -> ContinueState:
        next_count = prev_state.current_count + 1
        output_directory = self._determine_next_output_directory(
            continue_count=next_count,
            prev_setting=prev_state.setting,
        )
        setting = self._create_setting(
            current_count=next_count,
            prev_setting=prev_state.setting,
            prev_output_directory=prev_state.output_directory,
        )
        return ContinueState(
            current_count=next_count,
            output_directory=output_directory,
            setting=setting,
            preset_trainer=None,
        )

    def _determine_next_output_directory(
        self, continue_count: int, prev_setting: PhlowerSetting
    ) -> pathlib.Path:
        suffix = self._param_updator.get_output_directory_name_suffix(
            prev_setting=prev_setting.training,
            continue_count=continue_count,
        )

        base_directory = self._cont0_output_directory.parent
        base_name = self._cont0_output_directory.name
        return base_directory / f"{base_name}{suffix}"

    def _create_setting(
        self,
        current_count: int,
        prev_setting: PhlowerSetting,
        prev_output_directory: pathlib.Path,
    ) -> PhlowerSetting:

        _new_trainer_setting = self._param_updator.update_parameters(
            prev_setting=prev_setting.training,
            continue_count=current_count,
        )

        # Updating pretrained_directory is common
        new_trainer_setting = type(_new_trainer_setting).model_validate(
            _new_trainer_setting.model_dump()
            | {
                "initializer_setting": {
                    "type_name": "pretrained",
                    "reference_directory": prev_output_directory,
                }
            }
        )

        # NOTE: It is necessary to dump and load the setting
        # because validation is not performed when using model_copy.
        return PhlowerSetting.model_validate(
            prev_setting.model_dump()
            | {"training": new_trainer_setting.model_dump()}
        )


class PhlowerContinueTrainer:
    def __init__(
        self,
        setting: PhlowerSetting,
        *,
        current_count: int = 0,
        cont0_output_directory: pathlib.Path | None = None,
        trainer: PhlowerTrainer | None = None,
        prev_output_directory: pathlib.Path | None = None,
        promote_next_state: bool = False,
    ):
        self._cont0_output_directory: pathlib.Path | None = (
            cont0_output_directory
        )

        # NOTE: This attribute is updated during training
        self._state = ContinueState(
            current_count=current_count,
            output_directory=prev_output_directory,
            setting=setting,
            preset_trainer=trainer,
        )
        self._renderer: Callable[[pathlib.Path], None] | None = None
        self._extra_handlers: dict[str, tuple[IPhlowerHandler, bool]] = {}
        self._promote_next_state = promote_next_state

    @classmethod
    def restart_from(
        cls,
        model_directory: pathlib.Path,
        decrypt_key: bytes | None = None,
    ) -> PhlowerContinueTrainer:

        status = _retrieve_continue_status(model_directory)
        if status is None:
            _logger.info(
                "Cannot retrieve continue status from model directory. "
                "Assuming this is the first run of training."
            )
            trainer = PhlowerTrainer.restart_from(
                model_directory=model_directory,
                decrypt_key=decrypt_key,
            )
            return PhlowerContinueTrainer(
                setting=trainer._setting,
                trainer=trainer,
            )

        current_count, cont0_output_directory = status
        try:
            trainer = PhlowerTrainer.restart_from(
                model_directory=model_directory,
                decrypt_key=decrypt_key,
            )
            return PhlowerContinueTrainer(
                setting=trainer._setting,
                current_count=current_count,
                cont0_output_directory=cont0_output_directory,
                trainer=trainer,
            )
        except PhlowerRestartTrainingCompletedError as ex:
            _logger.info(
                "Training is already completed. "
                "Try to increment the continue_count and restart training."
            )
            setting = PhlowerSetting.read_yaml(
                model_directory / "model.yml", decrypt_key=decrypt_key
            )
            if current_count == setting.training.continue_setting.stop_count:
                raise PhlowerRestartTrainingCompletedError(
                    "Continue training is also completed."
                ) from ex

            # create initialized trainer
            setting = PhlowerSetting.rewrite_model_initializer(
                reference_setting=setting,
                initializer_type="pretrained",
                reference_directory=model_directory,
            )
            trainer = PhlowerTrainer.from_setting(
                setting, decrypt_key=decrypt_key
            )
            return PhlowerContinueTrainer(
                setting=trainer._setting,
                current_count=current_count,
                cont0_output_directory=cont0_output_directory,
                trainer=trainer,
                promote_next_state=True,
                prev_output_directory=model_directory,
            )

    @classmethod
    def from_setting(
        cls,
        setting: PhlowerSetting,
        decrypt_key: bytes | None = None,
    ) -> PhlowerContinueTrainer:

        init_setting = setting.training.initializer_setting
        if init_setting.type_name == TrainerInitializeType.restart:
            trainer = PhlowerContinueTrainer.restart_from(
                init_setting.reference_directory, decrypt_key=decrypt_key
            )
            return trainer

        setting.model.resolve()
        trainer = PhlowerContinueTrainer(setting)
        return trainer

    def set_trainer_renderer(
        self, renderer: Callable[[pathlib.Path], None]
    ) -> None:
        self._renderer = renderer

    def attach_handler(
        self, name: str, handler: IPhlowerHandler, allow_overwrite: bool = False
    ) -> None:
        """
        Attach extra handlers to the trainer.
        It is useful when you want to use your own handler which is
        initialized outside of training. (e.g. Optuna, WandB, etc.)
        """
        self._extra_handlers[name] = (handler, allow_overwrite)

    @overload
    def train(
        self,
        output_directory: pathlib.Path,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ) -> None:
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
    ) -> None:
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

        """

    def get_registered_setting(self) -> PhlowerSetting:
        return self._state.setting

    def train(
        self,
        output_directory: pathlib.Path,
        train_directories: list[pathlib.Path] | None = None,
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ) -> None:
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
            None
        """

        self._cont0_output_directory = self._resolve_cont0_output_directory(
            output_directory
        )
        state_updater = StateUpdator(
            cont0_output_directory=self._cont0_output_directory,
            continue_setting=self._state.setting.training.continue_setting,
        )
        self._state = self._resolve_state(
            state_updator=state_updater, output_directory=output_directory
        )
        # NOTE: The first training should be done
        # in case restarting from a previous checkpoint.
        _ = self._train(
            train_directories=train_directories,
            validation_directories=validation_directories,
            disable_dimensions=disable_dimensions,
            decrypt_key=decrypt_key,
            encrypt_key=encrypt_key,
        )

        if self._state.setting.training.continue_setting.is_active is False:
            _logger.info(
                "No continue_setting found. "
                "Training completed after the first run."
            )
            return

        current_count = self._state.current_count
        end_count = self._state.setting.training.continue_setting.stop_count

        for _ in range(current_count + 1, end_count + 1):
            self._state = state_updater.update(prev_state=self._state)
            _ = self._train(
                train_directories=train_directories,
                validation_directories=validation_directories,
                disable_dimensions=disable_dimensions,
                decrypt_key=decrypt_key,
                encrypt_key=encrypt_key,
            )

        return None

    def _resolve_cont0_output_directory(
        self, output_directory: pathlib.Path
    ) -> pathlib.Path:

        if self._state.current_count == 0:
            return output_directory

        if self._cont0_output_directory is None:
            raise ValueError(
                "cont0_output_directory is not set. "
                "Cannot continue training without cont0_output_directory."
                "It seems that the trainer is failed to restore "
                "the continue state from the model directory. "
            )

        if self._cont0_output_directory.parent != output_directory.parent:
            _logger.info(
                "The parent directory of the output_directory must be the "
                "same as the parent directory of cont0_output_directory."
                "This may be caused by restarting training in a different "
                "directory than the original training."
                f"cont0_output_directory: {self._cont0_output_directory}, "
                f"output_directory: {output_directory}",
            )
            _logger.info(
                "cont0_output_directory will be set as "
                "output_directory.parent / cont0_output_directory.name."
            )
            return output_directory.parent / self._cont0_output_directory.name

        return self._cont0_output_directory

    def _resolve_state(
        self, state_updator: StateUpdator, output_directory: pathlib.Path
    ) -> ContinueState:
        if not self._promote_next_state:
            _state = self._state.initialize_output_directory(output_directory)
            return _state

        # This is a following case:
        # model_cont_3 is completely finished,
        # and the next training is model_cont_4.
        if output_directory == self._state.output_directory:
            return state_updator.update(prev_state=self._state)

        _tmp = state_updator.update(prev_state=self._state)
        return _tmp.initialize_output_directory(
            output_directory.parent / _tmp.output_directory.name
        )

    def _train(
        self,
        train_directories: list[pathlib.Path] | None = None,
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ):
        if self._state.current_count > 0:
            self._save_continue_state(self._state)

        trainer = self._state.setup_trainer(decrypt_key=decrypt_key)
        trainer.update_handlers_activity(
            continue_count=self._state.current_count
        )
        if self._renderer is not None:
            trainer.draw_model(self._state.output_directory)
            self._renderer(self._state.output_directory)
        for name, (handler, allow_overwrite) in self._extra_handlers.items():
            trainer.attach_handler(
                name, handler, allow_overwrite=allow_overwrite
            )

        _start_training(
            trainer=trainer,
            train_directories=train_directories,
            validation_directories=validation_directories,
            output_directory=self._state.output_directory,
            decrypt_key=decrypt_key,
            encrypt_key=encrypt_key,
            disable_dimensions=disable_dimensions,
        )

    def _save_continue_state(self, state: ContinueState) -> None:
        content = {
            "continue_count": state.current_count,
            "cont0_output_directory": str(self._cont0_output_directory),
        }
        if (state.output_directory / "continue_state.yml").exists():
            _logger.info(
                "continue_state.yml already exists. "
                "Overwriting the file with the new state."
            )
            return

        _ = PhlowerYamlFile.save(
            output_directory=state.output_directory,
            file_basename="continue_state",
            data=content,
            encrypt_key=None,
        )


def _start_training(
    trainer: PhlowerTrainer,
    output_directory: pathlib.Path,
    train_directories: list[pathlib.Path] | None = None,
    validation_directories: list[pathlib.Path] | None = None,
    disable_dimensions: bool = False,
    encrypt_key: bytes | None = None,
    decrypt_key: bytes | None = None,
) -> None:
    setting = trainer.get_registered_trainer_setting()

    if setting.parallel_setting.is_active:
        _start_parallel_training(
            trainer=trainer,
            output_directory=output_directory,
            train_directories=train_directories,
            validation_directories=validation_directories,
            disable_dimensions=disable_dimensions,
            decrypt_key=decrypt_key,
            encrypt_key=encrypt_key,
        )

    else:
        _logger.info("Start single training.")
        trainer.train(
            train_directories=train_directories,
            validation_directories=validation_directories,
            output_directory=output_directory,
            disable_dimensions=disable_dimensions,
            decrypt_key=decrypt_key,
            encrypt_key=encrypt_key,
        )


# NOTE: error handler should be called here, not in PhlowerTrainer.train,
# because the error handler should be called
# even if the error occurs in the parallel training process.
@error_handler_wrapper("trainer")
def _start_parallel_training(
    trainer: PhlowerTrainer,
    output_directory: pathlib.Path,
    train_directories: list[pathlib.Path] | None = None,
    validation_directories: list[pathlib.Path] | None = None,
    disable_dimensions: bool = False,
    encrypt_key: bytes | None = None,
    decrypt_key: bytes | None = None,
) -> None:
    setting = trainer.get_registered_trainer_setting()

    if setting.parallel_setting.parallel_type != "DDP":
        raise NotImplementedError(
            "Only DDP is supported for continue training."
        )
    world_size = setting.parallel_setting.world_size
    _logger.info(f"Start parallel training with world_size={world_size}.")
    mp.spawn(
        partial(
            trainer.train_ddp,
            train_directories=train_directories,
            validation_directories=validation_directories,
            disable_dimensions=disable_dimensions,
            decrypt_key=decrypt_key,
            encrypt_key=encrypt_key,
        ),
        args=(
            world_size,
            output_directory,
        ),
        nprocs=world_size,
        join=True,
    )


def _retrieve_continue_status(
    model_directory: pathlib.Path,
) -> tuple[int, pathlib.Path] | None:
    continue_state_file = model_directory / "continue_state.yml"
    if continue_state_file.exists():
        continue_state = PhlowerYamlFile(continue_state_file).load()
        current_count = int(continue_state["continue_count"])
        cont0_output_directory = pathlib.Path(
            continue_state["cont0_output_directory"]
        )
        return current_count, cont0_output_directory

    _logger.info(
        "No continue_state.yml found in the model directory. "
        "Try to retrieve status from output directory name."
    )

    # No continue_state.yml found
    # try to retieve from output directory name
    # This is for backward compatibility with old version of Phlower
    items: list[str] = model_directory.name.split("_")
    if len(items) < 2:
        return None
    if not items[-2].startswith("cont"):
        return None

    num = items[-2].removeprefix("cont")
    return int(num), model_directory.parent / "_".join(items[:-2])
