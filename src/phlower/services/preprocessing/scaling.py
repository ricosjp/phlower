from __future__ import annotations

import pathlib
from functools import partial

from phlower.io import PhlowerDirectory
from phlower.io._files import PhlowerNumpyFile
from phlower.services.preprocessing import ScalersComposition
from phlower.settings import PhlowerScalingSetting
from phlower.utils import get_logger
from phlower.utils._multiprocessor import PhlowerMultiprocessor
from phlower.utils.typing import ArrayDataType

logger = get_logger(__name__)


class ScalingService:
    """
    This is Facade Class for scaling process
    """

    @classmethod
    def from_pickle(
        cls,
        pickle_file_path: pathlib.Path,
        decrypt_key: bytes = None,
    ):
        scalers = ScalersComposition.from_pickle_file(
            pickle_file_path, decrypt_key=decrypt_key
        )
        return ScalingService(scalers)

    @classmethod
    def from_setting(
        cls,
        scaling_setting: PhlowerScalingSetting,
    ):
        _scalers = ScalersComposition.from_setting(scaling_setting)
        return ScalingService(scalers=_scalers)

    def __init__(
        self,
        scalers: ScalersComposition,
    ) -> None:
        self._scalers = scalers

    def fit_transform(self) -> None:
        """This function is consisted of these three process.
        - Determine parameters of scalers by reading data files lazily
        - Transform interim data and save result
        - Save file of parameters

        Returns
        -------
        None
        """
        self.lazy_fit_all()
        self.transform_interim()
        self.save()

    def lazy_fit_all(
        self, scaler_name_to_files: dict[str, list[pathlib.Path]]
    ) -> None:
        """Determine preprocessing parameters
        by reading data files lazily.

        Returns
        -------
        None
        """
        scaler_name_to_files = {
            s: self._setting.collect_fitting_files(s)
            for s in self._scalers.get_scaler_names()
        }
        self._scalers.lazy_partial_fit(scaler_name_to_files)

    def transform_interim(
        self,
        setting: PhlowerScalingSetting,
        *,
        max_process: int = None,
        allow_missing: bool = False,
        force_renew: bool = False,
        decrypt_key: bytes | None = None,
    ) -> None:
        """
        Apply scaling process to data in interim directory and save results
        in preprocessed directory.

        Parameters
        ----------
            group_id: int, optional
                group_id to specify chunk of preprocessing group. Useful when
                MemoryError occurs with all variables preprocessed in one node.
                If not specified, process all variables.

        Returns
        -------
        None
        """

        interim_dirs = setting.collect_interim_directories()
        variable_names = setting.get_variable_names()

        processor = PhlowerMultiprocessor(max_process=max_process)
        processor.run(
            variable_names,
            target_fn=partial(
                self._transform_directories,
                setting=setting,
                directories=interim_dirs,
                allow_missing=allow_missing,
                force_renew=force_renew,
                decrypt_key=decrypt_key,
            ),
            chunksize=1,
        )

    def inverse_transform(
        self, dict_data: dict[str, ArrayDataType]
    ) -> dict[str, ArrayDataType]:
        return self._scalers.inverse_transform(dict_data)

    def save(
        self, pickle_file_path: pathlib.Path, encrypt_key: bytes = None
    ) -> None:
        """
        Save Parameters of scaling converters
        """
        self._scalers.save(
            pickle_file_path=pickle_file_path, encrypt_key=encrypt_key
        )

    def _transform_directories(
        self,
        setting: PhlowerScalingSetting,
        variable_name: str,
        directories: list[pathlib.Path],
        allow_missing: bool = False,
        force_renew: bool = False,
        decrypt_key: bytes | None = None,
    ) -> None:
        for path in directories:
            output_dir = PhlowerDirectory(setting.get_output_directory(path))
            if self._can_skip(output_dir, variable_name, force_renew):
                return

            siml_file = PhlowerDirectory(path).find_variable_file(
                variable_name, allow_missing=allow_missing
            )
            if siml_file is None:
                logger.warning(
                    f"Scaling skipped. {variable_name} is missing in {path}"
                )
                return

            transformed_data = self._scalers.transform_file(
                variable_name, siml_file, decrypt_key=decrypt_key
            )

            PhlowerNumpyFile.save_variables(
                output_directory=output_dir,
                file_basename=variable_name,
                data=transformed_data,
                encrypt_key=decrypt_key,
            )

    def _can_skip(
        self,
        output_dir: PhlowerDirectory,
        variable_name: str,
        force_renew: bool,
    ) -> bool:
        if force_renew:
            return False

        if output_dir.exist_variable_file(variable_name):
            logger.info(
                f"{output_dir.path} / {variable_name} "
                "already exists. Skipped."
            )
            return True

        return False
