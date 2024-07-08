from __future__ import annotations

import pathlib
from functools import partial

from pipe import chain, select, where

from phlower.io import PhlowerDirectory, PhlowerFileBuilder
from phlower.io._files import IPhlowerNumpyFile, PhlowerNumpyFile
from phlower.services.preprocessing import ScalersComposition
from phlower.services.preprocessing._scalers import ScalerWrapper
from phlower.settings import PhlowerScalingSetting, ScalerResolvedParameter
from phlower.utils import get_logger
from phlower.utils._multiprocessor import PhlowerMultiprocessor
from phlower.utils.typing import ArrayDataType

logger = get_logger(__name__)


class ScalingService:
    """
    This is Facade Class for scaling process
    """

    def __init__(
        self,
        scaling_setting: PhlowerScalingSetting,
    ) -> None:
        self._scaling_setting = scaling_setting
        self._parameters = scaling_setting.resolve_scalers()
        self._scalers = ScalersComposition.from_setting(self._parameters)

    def fit_transform(
        self,
        interim_data_directories: list[pathlib.Path],
        decrypt_key: bytes | None = None,
    ) -> None:
        """This function is consisted of these three process.
        - Determine parameters of scalers by reading data files lazily
        - Transform interim data and save result
        - Save file of parameters

        Returns
        -------
        None
        """
        self.lazy_fit_all(
            data_directories=interim_data_directories, decrypt_key=decrypt_key
        )
        self.transform_interim(
            data_directories=interim_data_directories,
            output_base_directory=self._scaling_setting.output_directory,
            decrypt_key=decrypt_key,
        )
        save_file_path = (
            self._scaling_setting.output_directory / "preprocess.yml"
        )
        self.save(save_file_path)

    def lazy_fit_all(
        self,
        data_directories: list[pathlib.Path],
        *,
        max_process: int = None,
        decrypt_key: bytes | None = None,
    ) -> None:
        processor = PhlowerMultiprocessor(max_process=max_process)
        results = processor.run(
            self._parameters,
            target_fn=partial(
                self._lazy_fit,
                data_directories=data_directories,
                decrypt_key=decrypt_key,
            ),
            chunksize=1,
        )

        # NOTE: When using multiprocessing, parameters of each scaler are
        #  updated in each process. However, it is not reflected in parent process.
        #  because we do not share memory.
        # Thus, in this implementation, recreate scalers compositions

        self._scalers.force_update(dict(results))
        return

    def _lazy_fit(
        self,
        parameter: ScalerResolvedParameter,
        data_directories: list[pathlib.Path],
        decrypt_key: bytes | None = bytes,
    ) -> tuple[str, ScalerWrapper]:
        fitting_files = list(
            data_directories
            | select(lambda x: parameter.collect_fitting_files(x))
            | chain
        )
        scaler_name = parameter.scaler_name
        self._scalers.lazy_partial_fit(
            scaler_name, fitting_files, decrypt_key=decrypt_key
        )
        return scaler_name, self._scalers.get_scaler(scaler_name)

    def transform_interim(
        self,
        data_directories: list[pathlib.Path],
        output_base_directory: pathlib.Path,
        *,
        max_process: int = None,
        allow_missing: bool = False,
        allow_overwrite: bool = False,
        decrypt_key: bytes | None = None,
    ) -> None:
        """
        Apply scaling process to data in interim directory and save results
        in preprocessed directory.

        Returns
        -------
        None
        """

        processor = PhlowerMultiprocessor(max_process=max_process)
        processor.run(
            self._parameters,
            target_fn=partial(
                self._transform_directories,
                data_directories=data_directories,
                output_base_directory=output_base_directory,
                allow_missing=allow_missing,
                allow_overwrite=allow_overwrite,
                decrypt_key=decrypt_key,
            ),
            chunksize=1,
        )

    def _transform_directories(
        self,
        parameter: ScalerResolvedParameter,
        data_directories: list[pathlib.Path],
        output_base_directory: pathlib.Path,
        allow_missing: bool = False,
        allow_overwrite: bool = False,
        decrypt_key: bytes | None = None,
    ) -> None:
        transform_files: list[IPhlowerNumpyFile] = list(
            data_directories
            | select(
                lambda x: parameter.collect_transform_files(
                    x, allow_missing=allow_missing
                )
            )
            | chain
        )

        for numpy_file in transform_files:
            # HACK: Apply path rule
            output_directory = (
                output_base_directory / numpy_file.file_path.parent
            )

            transformed_data = self._scalers.transform_file(
                parameter.scaler_name, numpy_file, decrypt_key=decrypt_key
            )

            PhlowerNumpyFile.save(
                output_directory=output_directory,
                file_basename=numpy_file.get_variable_name(),
                data=transformed_data,
                encrypt_key=decrypt_key,
                allow_overwrite=allow_overwrite,
            )

    def inverse_transform(
        self,
        dict_data: dict[str, ArrayDataType],
        raise_missing_message: bool = False,
    ) -> dict[str, ArrayDataType]:
        _filtered = self._filter_scalable_variables(
            dict_data.keys(), raise_missing_message=raise_missing_message
        )
        return {
            name: self._scalers.inverse_transform(scaler_name, dict_data[name])
            for name, scaler_name in _filtered
        }

    def _filter_scalable_variables(
        self, variable_names: list[str], raise_missing_message: bool = False
    ) -> list[tuple[str, str]]:
        _filtered: list[tuple[str, str]] = list(
            variable_names
            | where(lambda x: self._scaling_setting.is_scaler_exist(x))
            | select(lambda x: (x, self._scaling_setting.get_scaler_name(x)))
        )

        if not raise_missing_message:
            return _filtered

        for v in variable_names:
            if not self._scaling_setting.is_scaler_exist(v):
                logger.warning(f"scaler for {v} is not found.")

        return _filtered

    def save(
        self, save_file_path: pathlib.Path, encrypt_key: bytes = None
    ) -> None:
        """
        Save Parameters of scaling converters
        """

        scalers_data = self._scalers.get_dumped_data()

        _dumped_scalers: dict = {}
        for name in self._scaling_setting.get_variable_names():
            _setting = self._scaling_setting.varaible_name_to_scalers[name]
            if _setting.is_parent_scaler:
                _dumped_scalers[name] = scalers_data[
                    _setting.get_scaler_name(name)
                ]
            else:
                _dumped_scalers[name] = _setting

        dump_setting = PhlowerScalingSetting(
            varaible_name_to_scalers=_dumped_scalers
        )

        yaml_file = PhlowerFileBuilder.yaml_file(save_file_path)
        yaml_file.save(dump_setting.model_dump(), overwrite=True)
