from __future__ import annotations

import datetime as dt
import gc
import warnings
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

from phlower.utils.typing import ArrayDataType
from phlower.utils.enums import PhlowerFileExtType
from phlower.services.preprocessing import IPhlowerScaler
from phlower.services.preprocessing._scalers import scale_functions
from phlower.io._files import IPhlowerNumpyFile
from phlower._base.array import phlower_array

from phlower.settings import ScalerParameters

# HACK: dict_data やめられるか？、中身わからん不安

class ScalerWrapper(IPhlowerScaler):
    @classmethod
    def from_setting(
        cls,
        setting: ScalerParameters
    ) -> ScalerWrapper:
        _cls = cls(
            method=setting.method,
            componentwise=setting.component_wise,
            **setting.parameters
        )

        # After initialization, set other private properties such as var_, max_
        for k, v in setting.parameters.items():
            setattr(_cls.converter, k, v)
        return _cls

    def __init__(
        self,
        method_name: str,
        *,
        componentwise: bool = True,
        parameters: dict = None
    ):

        self.method_name = method_name
        self.converter = scale_functions.create_scaler(
            method_name,
            **parameters
        )
        self.componentwise = componentwise

    @property
    def use_diagonal(self) -> bool:
        return self.converter.use_diagonal

    def is_erroneous(self) -> bool:
        return self.converter.is_erroneous()

    def partial_fit(
        self,
        data: ArrayDataType
    ) -> None:
        wrapped_data = phlower_array(data)
        reshaped_data = wrapped_data.reshape(
            componentwise=self.componentwise,
            skip_nan=True,
            use_diagonal=self.use_diagonal
        )
        if reshaped_data.size == 0:
            warnings.warn(
                "Found array with 0 sample(s) after deleting nan items"
            )
            return

        self.converter.partial_fit(reshaped_data)
        return

    def transform(
        self,
        data: ArrayDataType
    ) -> Union[np.ndarray, sp.coo_matrix]:

        wrapped_data = phlower_array(data)
        result = wrapped_data.apply(
            self.converter.transform,
            componentwise=self.componentwise,
            skip_nan=False,
            use_diagonal=False
        )
        return result

    def inverse_transform(
        self,
        data: ArrayDataType
    ) -> Union[np.ndarray, sp.coo_matrix]:
        wrapped_data = phlower_array(data)
        result = wrapped_data.apply(
            self.converter.inverse_transform,
            componentwise=self.componentwise,
            skip_nan=False,
            use_diagonal=False
        )
        return result

    def lazy_partial_fit(
        self,
        data_files: list[IPhlowerNumpyFile]
    ) -> None:
        for data_file in data_files:
            data = self._load_file(data_file)
            self.partial_fit(data)
            # del data
            # gc.collect()
        return

    def get_dumped_data(self) -> ScalerParameters:
        dumped_dict = {
            'method': self.method_name,
            'componentwise': self.componentwise,
            'parameters': vars(self.converter)
        }
        return ScalerParameters(**dumped_dict)

    def _load_file(
        self,
        siml_file: IPhlowerNumpyFile,
        decrypt_key: bytes | None = None
    ) -> ArrayDataType:

        loaded_data = siml_file.load(decrypt_key=decrypt_key)

        if siml_file.file_extension in [
            PhlowerFileExtType.NPZENC.value,
            PhlowerFileExtType.NPZ.value
        ]:
            if not sp.issparse(loaded_data):
                raise ValueError(
                    f"Data type not understood for: {siml_file.file_path}"
                )

        return loaded_data
