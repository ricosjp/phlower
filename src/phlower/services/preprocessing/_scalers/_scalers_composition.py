from __future__ import annotations
import multiprocessing as multi
import warnings
import pathlib
from typing import Optional

from phlower.services.preprocessing import ScalerWrapper, IPhlowerScaler
from phlower.utils.typing import ArrayDataType
from phlower.io._files import PhlowerNumpyFile
from phlower.settings import PhlowerScalingSetting

from phlower.utils import get_logger, determine_n_process
from phlower.services.preprocessing._scalers._scaler_loader import ScalerFileIO


logger = get_logger(__name__)


class ScalersComposition(IPhlowerScaler):

    @classmethod
    def from_pickle_file(
        cls,
        pkl_file_path: pathlib.Path,
        max_process: Optional[int] = None,
        decrypt_key: Optional[bytes] = None
    ) -> ScalersComposition:

        variable_name_to_scalers, scalers_dict = \
            ScalerFileIO.load_pickle(pkl_file_path, decrypt_key=decrypt_key)

        return cls(
            variable_name_to_scalers=variable_name_to_scalers,
            scalers_dict=scalers_dict,
            max_process=max_process
        )

    @classmethod
    def from_setting(
        cls,
        setting: PhlowerScalingSetting,
        max_process: Optional[int] = None,
    ) -> ScalersComposition:
        variable_name_to_scalers = _init_scalers_relationship(setting)
        scalers_dict = {
            name: ScalerWrapper.from_setting(v)
            for name, v in setting.name_to_scalers.items()
            if v.same_as is None
        }
        return cls(
            variable_name_to_scalers=variable_name_to_scalers,
            scalers_dict=scalers_dict,
            max_process=max_process
        )

    def __init__(
        self,
        variable_name_to_scalers: dict[str, str],
        scalers_dict: dict[str, ScalerWrapper],
        max_process: Optional[int] = None
    ) -> None:

        self._n_process = determine_n_process(max_process)

        # varaible name to saler name
        # scaler name corresponds with variable name except same_as
        self._variable_name_to_scaler = variable_name_to_scalers

        # variable_name to siml_scelar
        self._scalers_dict = scalers_dict

    def get_variable_names(self, group_id: Optional[int] = None) -> list[str]:
        variable_names = list(self._variable_name_to_scaler.keys())
        if group_id is None:
            return variable_names

        variable_names = [
            k for k in variable_names
            if self.get_scaler(k).group_id == group_id
        ]
        return variable_names

    def get_scaler_names(self, group_id: Optional[int] = None) -> list[str]:
        scaler_names = list(self._scalers_dict.keys())
        if group_id is None:
            return scaler_names

        scaler_names = [
            k for k in scaler_names
            if self._scalers_dict[k].group_id == group_id
        ]
        return scaler_names

    def get_scaler(
        self,
        variable_name: str,
        allow_missing: bool = False
    ) -> ScalerWrapper | None:
        scaler_name = self._variable_name_to_scaler.get(variable_name)
        scaler = self._scalers_dict.get(scaler_name)
        if allow_missing:
            return scaler

        if scaler is None:
            raise ValueError(
                f"No Scaler for {scaler_name} is found."
            )
        return scaler

    def save(self, pickle_file_path: pathlib.Path, encrypt_key: bytes | None = None) -> None:
        scalers_dict = {
            k: self.get_scaler(k).get_dumped_dict()
            for k in self.get_scaler_names()
        }

        fileio = ScalerFileIO()
        fileio.save_pickle(
            pkl_file_path=pickle_file_path,
            variable_name_to_scalers=self._variable_name_to_scaler,
            scalers_dict=scalers_dict,
            encrypt_key=encrypt_key
        )

    def lazy_partial_fit(
        self,
        scaler_name_to_files: dict[str, list[PhlowerNumpyFile]]
    ) -> None:
        preprocessor_inputs: list[tuple[str, list[PhlowerNumpyFile]]] \
            = [
                (name, files)
                for name, files in scaler_name_to_files.items()
        ]

        with multi.Pool(self._max_process) as pool:
            results = pool.starmap(
                self._lazy_partial_fit,
                preprocessor_inputs,
                chunksize=1
            )
        for name, scaler in results:
            self._scalers_dict[name] = scaler

    def transform(
        self,
        variable_name: str,
        data: ArrayDataType
    ) -> ArrayDataType:
        scaler = self.get_scaler(variable_name)
        transformed_data = scaler.transform(data)
        return transformed_data

    def transform_file(
        self,
        variable_name: str,
        siml_file: PhlowerNumpyFile,
        decrypt_key: bytes | None = None
    ) -> ArrayDataType:

        loaded_data = siml_file.load(decrypt_key=decrypt_key)
        scaler = self.get_scaler(variable_name)
        transformed_data = scaler.transform(loaded_data)
        return transformed_data

    def transform_dict(
        self,
        dict_data: dict[str, ArrayDataType],
        raise_missing_warning: bool = True
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
        self,
        variable_name: str,
        data: ArrayDataType
    ) -> ArrayDataType:
        scaler = self.get_scaler(variable_name)
        return scaler.inverse_transform(data)

    def inverse_transform_dict(
        self,
        dict_data: dict[str, ArrayDataType],
        raise_missing_warning: bool = True
    ) -> dict[str, ArrayDataType]:

        converted_dict_data: dict[str, ArrayDataType] = {}
        for variable_name, data in dict_data.items():
            scaler = self.get_scaler(variable_name, allow_missing=True)
            if scaler is None:
                if raise_missing_warning:
                    warnings.warn(
                        f"Scaler for {variable_name} is not found. Skipped"
                    )
                continue

            converted_data = scaler.inverse_transform(data)
            converted_dict_data[variable_name] = converted_data
        return converted_dict_data

    def _lazy_partial_fit(
        self,
        variable_name: str,
        data_files: list[PhlowerNumpyFile]
    ) -> tuple[str, ScalerWrapper]:
        scaler = self.get_scaler(variable_name)

        scaler.lazy_partial_fit(data_files)
        return (variable_name, scaler)



def _init_scalers_relationship(
    setting: PhlowerScalingSetting
) -> dict[str, str]:

    _dict = {}
    for variable_name, p_setting in setting.name_to_scalers.items():
        parent = p_setting.same_as
        if parent is None:
            _dict[variable_name] = f"scaler_{variable_name}"
        else:
            # same_as is set so no need to prepare preprocessor
            _dict[variable_name] = f"scaler_{parent}"
    return _dict
