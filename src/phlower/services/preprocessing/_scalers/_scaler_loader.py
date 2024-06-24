import pathlib

from phlower.io import PhlowerFileBuilder
from typing import Optional, Any

from phlower.services.preprocessing import ScalerWrapper
from phlower.services.preprocessing._scalers._scaler_parameters import ScalerParameters



class ScalerFileIO:
    def __init__(self) -> None:
        self._name_map_key = "variable_name_to_scalers"
        self._scalers_key = "scalers"

    def load_pickle(
        self,
        pkl_file_path: pathlib.Path,
        decrypt_key: bytes | None = None
    ) -> tuple[dict[str, str], dict[str, ScalerWrapper]]:

        pickle_file = PhlowerFileBuilder.pickle_file(pkl_file_path)
        _parameters: dict[str, dict] = pickle_file.load(decrypt_key=decrypt_key)
        variable_name_to_scalers = _parameters[self._name_map_key]

        scalers_dict: dict[str, ScalerWrapper] = {}
        for name, params in _parameters[self._scalers_key].items():
            scalers_dict[name] = ScalerWrapper.create(params)
        
        return variable_name_to_scalers, scalers_dict

    def save_pickle(
        self,
        pkl_file_path: pathlib.Path | str,
        variable_name_to_scalers: dict[str, str],
        scalers_dict: dict[str, Any],
        encrypt_key: bytes | None = None
    ) -> None:
        dict_data = {
            self._scalers_key: scalers_dict,
            self._name_map_key: variable_name_to_scalers
        }
        pickle_file = PhlowerFileBuilder.pickle_file(pkl_file_path)
        pickle_file.save(dict_data, overwrite=True, encrypt_key=encrypt_key)
