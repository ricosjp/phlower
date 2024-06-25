from __future__ import annotations

import multiprocessing as multi
import pathlib
from typing import Optional

from phlower.io._files import PhlowerNumpyFile
from phlower.services.preprocessing._scalers import (
    IPhlowerScaler,
    ScalerWrapper,
)
from phlower.services.preprocessing._scalers._scaler_loader import ScalerFileIO
from phlower.settings import PhlowerScalingSetting
from phlower.utils import determine_n_process, get_logger
from phlower.utils.typing import ArrayDataType

logger = get_logger(__name__)


class ScalersComposition(IPhlowerScaler):

    @classmethod
    def from_pickle_file(
        cls, pkl_file_path: pathlib.Path, decrypt_key: Optional[bytes] = None
    ) -> ScalersComposition:

        variable_name_to_scaler, scalers_dict = ScalerFileIO.load_pickle(
            pkl_file_path, decrypt_key=decrypt_key
        )

        return ScalersComposition(
            variable_name_to_scaler=variable_name_to_scaler,
            scalers_dict=scalers_dict
        )

    @classmethod
    def from_setting(cls, setting: PhlowerScalingSetting) -> ScalersComposition:
        variable_name_to_scaler = {
            k: setting.get_scaler_name(k)
            for k in setting.get_variable_names()
        }

        scalers_dict = {
            k: ScalerWrapper.from_setting(v)
            for k, v in setting.get_effective_scaler_parameters().items()
        }

        return cls(
            variable_name_to_scaler=variable_name_to_scaler,
            scalers_dict=scalers_dict,
        )

    def __init__(
        self,
        variable_name_to_scaler: dict[str, str],
        scalers_dict: dict[str, ScalerWrapper],
    ) -> None:
        self._variable_name_to_scaler_name = variable_name_to_scaler
        self._scalers_dict = scalers_dict

    def get_scaler_names(self) -> list[str]:
        scaler_names = list(self._scalers_dict.keys())
        return scaler_names

    def get_scaler(
        self, variable_name: str, allow_missing: bool = False
    ) -> ScalerWrapper | None:
        scaler_name = self._variable_name_to_scaler_name.get(variable_name)
        scaler = self._scalers_dict.get(scaler_name)
        if allow_missing:
            return scaler

        if scaler is None:
            raise ValueError(f"Scaler named {scaler_name} is not found.")
        return scaler

    def save(
        self, pickle_file_path: pathlib.Path, encrypt_key: bytes | None = None
    ) -> None:
        scalers_dict = {
            k: scaler.get_dumped_data()
            for k, scaler in self._scalers_dict.items()
        }

        fileio = ScalerFileIO()
        fileio.save_pickle(
            pkl_file_path=pickle_file_path,
            variable_name_to_scalers=self._variable_name_to_scaler_name,
            scalers_dict=scalers_dict,
            encrypt_key=encrypt_key,
        )

    def lazy_partial_fit(
        self,
        scaler_name_to_files: dict[str, list[PhlowerNumpyFile]],
        max_process: int = None,
    ) -> None:
        preprocessor_inputs: list[tuple[str, list[PhlowerNumpyFile]]] = [
            (name, files) for name, files in scaler_name_to_files.items()
        ]

        n_process = determine_n_process(max_process)
        with multi.Pool(n_process) as pool:
            results = pool.starmap(
                self._lazy_partial_fit, preprocessor_inputs, chunksize=1
            )
        for name, scaler in results:
            self._scalers_dict[name] = scaler

    def transform(
        self, variable_name: str, data: ArrayDataType
    ) -> ArrayDataType:
        scaler = self.get_scaler(variable_name)
        transformed_data = scaler.transform(data)
        return transformed_data

    def transform_file(
        self,
        variable_name: str,
        siml_file: PhlowerNumpyFile,
        decrypt_key: bytes | None = None,
    ) -> ArrayDataType:

        loaded_data = siml_file.load(decrypt_key=decrypt_key)
        scaler = self.get_scaler(variable_name)
        transformed_data = scaler.transform(loaded_data)
        return transformed_data

    def transform_dict(
        self,
        dict_data: dict[str, ArrayDataType],
        raise_missing_warning: bool = True,
    ) -> dict[str, ArrayDataType]:

        converted_dict_data: dict[str, ArrayDataType] = {}
        for variable_name, data in dict_data.items():
            scaler = self.get_scaler(variable_name, allow_missing=True)
            if scaler is None:
                if raise_missing_warning:
                    logger.warning(
                        f"Scaler for {variable_name} is not found. Skipped"
                    )
                continue

            converted_data = scaler.transform(data)
            converted_dict_data[variable_name] = converted_data

        return converted_dict_data

    def inverse_transform(
        self, variable_name: str, data: ArrayDataType
    ) -> ArrayDataType:
        scaler = self.get_scaler(variable_name)
        return scaler.inverse_transform(data)

    def inverse_transform_dict(
        self,
        dict_data: dict[str, ArrayDataType],
        raise_missing_warning: bool = True,
    ) -> dict[str, ArrayDataType]:

        converted_dict_data: dict[str, ArrayDataType] = {}
        for variable_name, data in dict_data.items():
            scaler = self.get_scaler(variable_name, allow_missing=True)
            if scaler is None:
                if raise_missing_warning:
                    logger.warning(
                        f"Scaler for {variable_name} is not found. Skipped"
                    )
                continue

            converted_data = scaler.inverse_transform(data)
            converted_dict_data[variable_name] = converted_data
        return converted_dict_data

    def _lazy_partial_fit(
        self, variable_name: str, data_files: list[PhlowerNumpyFile]
    ) -> tuple[str, ScalerWrapper]:
        scaler = self.get_scaler(variable_name)

        scaler.lazy_partial_fit(data_files)
        return (variable_name, scaler)
