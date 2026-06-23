from __future__ import annotations

import pathlib
from typing import Final, NamedTuple, overload

from phlower.io import PhlowerYamlFile
from phlower.services.trainer._continue_updater import (
    ContinueParameterUpdatorFactory,
)
from phlower.services.trainer._trainer import PhlowerTrainer
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
        assert self.output_directory is None, (
            "output_directory is already set. "
            "Cannot set output_directory again."
            "Use StateUpdater to update the state instead."
        )
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


class StateUpdater:
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
    ):
        self._cont0_output_directory: pathlib.Path | None = (
            cont0_output_directory
        )

        # NOTE: This attribute is updated during training
        self._state = ContinueState(
            current_count=current_count,
            output_directory=None,
            setting=setting,
            preset_trainer=trainer,
        )

    @classmethod
    def restart_from(
        cls,
        model_directory: pathlib.Path,
        decrypt_key: bytes | None = None,
    ) -> PhlowerContinueTrainer:

        try:
            trainer = PhlowerTrainer.restart_from(
                model_directory=model_directory,
                decrypt_key=decrypt_key,
            )
        except PhlowerRestartTrainingCompletedError as ex:
            raise NotImplementedError(
                "Restarting training from a completed training"
                " is not supported yet."
            ) from ex

        continue_state_file = model_directory / "continue_state.yml"
        if not continue_state_file.exists():
            # No continue_state.yml found, assume this is the first run
            return PhlowerContinueTrainer(
                setting=trainer._setting,
                trainer=trainer,
            )

        continue_state = PhlowerYamlFile(continue_state_file).load()
        current_count = int(continue_state["continue_count"])
        cont0_output_directory = pathlib.Path(
            continue_state["cont0_output_directory"]
        )
        return PhlowerContinueTrainer(
            setting=trainer._setting,
            current_count=current_count,
            cont0_output_directory=cont0_output_directory,
            trainer=trainer,
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

    @overload
    def train(
        self,
        output_directory: pathlib.Path,
        disable_dimensions: bool = False,
        random_seed: int | None = None,
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

        Returns:
            None
        """

    def train(
        self,
        output_directory: pathlib.Path,
        train_directories: list[pathlib.Path] | None = None,
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ) -> None:

        if self._cont0_output_directory is None:
            self._cont0_output_directory = output_directory

        if self._cont0_output_directory.parent != output_directory.parent:
            _logger.info(
                "The parent directory of the output_directory must be the same "
                "as the parent directory of the cont0_output_directory."
                "This may be caused by restarting training in a different "
                "directory than the original training."
                f"cont0_output_directory: {self._cont0_output_directory}, "
                f"output_directory: {output_directory}",
            )
            _logger.info(
                "cont0_output_directory will be set as "
                "output_directory.parent / cont0_output_directory.name."
            )
            self._cont0_output_directory = (
                output_directory.parent / self._cont0_output_directory.name
            )

        # set output_directory for the first training
        self._state = self._state.initialize_output_directory(output_directory)

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

        state_updater = StateUpdater(
            cont0_output_directory=self._cont0_output_directory,
            continue_setting=self._state.setting.training.continue_setting,
        )
        current_count = self._state.current_count
        end_count = self._state.setting.training.continue_setting.stop_count

        for _ in range(current_count + 1, end_count + 1):
            self._state = state_updater.update(prev_state=self._state)
            self._save_continue_state(self._state)
            _ = self._train(
                train_directories=train_directories,
                validation_directories=validation_directories,
                disable_dimensions=disable_dimensions,
                decrypt_key=decrypt_key,
                encrypt_key=encrypt_key,
            )

        return None

    def _train(
        self,
        train_directories: list[pathlib.Path] | None = None,
        validation_directories: list[pathlib.Path] | None = None,
        disable_dimensions: bool = False,
        decrypt_key: bytes | None = None,
        encrypt_key: bytes | None = None,
    ):
        try:
            trainer = self._state.setup_trainer(decrypt_key=decrypt_key)
            trainer.train(
                output_directory=self._state.output_directory,
                train_directories=train_directories,
                validation_directories=validation_directories,
                disable_dimensions=disable_dimensions,
                decrypt_key=decrypt_key,
                encrypt_key=encrypt_key,
            )
        except Exception as ex:
            # TODO: Accept user defined custom exception handler
            raise ex

    def _save_continue_state(self, state: ContinueState) -> None:
        content = {
            "continue_count": state.current_count,
            "cont0_output_directory": str(self._cont0_output_directory),
        }
        _ = PhlowerYamlFile.save(
            output_directory=state.output_directory,
            file_basename="continue_state",
            data=content,
            encrypt_key=None,
        )
