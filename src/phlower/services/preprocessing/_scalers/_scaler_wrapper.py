from __future__ import annotations

import scipy.sparse as sp

from phlower._base.array import phlower_array
from phlower.io._files import IPhlowerNumpyFile
from phlower.services.preprocessing._scalers import (
    IPhlowerScaler,
    scale_functions,
)
from phlower.settings import ScalerInputParameters, ScalerResolvedParameters
from phlower.utils import get_logger
from phlower.utils.enums import PhlowerFileExtType
from phlower.utils.typing import ArrayDataType

logger = get_logger(__name__)


class ScalerWrapper(IPhlowerScaler):
    @classmethod
    def from_setting(cls, setting: ScalerResolvedParameters) -> ScalerWrapper:
        _cls = cls(
            method=setting.method,
            componentwise=setting.component_wise,
            **setting.parameters,
        )

        # After initialization, set other private properties such as var_, max_
        for k, v in setting.parameters.items():
            setattr(_cls._scaler, k, v)
        return _cls

    def __init__(
        self,
        method_name: str,
        *,
        componentwise: bool = True,
        parameters: dict = None,
    ):
        self.method_name = method_name
        self._scaler = scale_functions.create_scaler(method_name, **parameters)
        self.componentwise = componentwise

    @property
    def use_diagonal(self) -> bool:
        return self._scaler.use_diagonal

    def is_erroneous(self) -> bool:
        return self._scaler.is_erroneous()

    def partial_fit(self, data: ArrayDataType) -> None:
        wrapped_data = phlower_array(data)
        reshaped_data = wrapped_data.reshape(
            componentwise=self.componentwise,
            skip_nan=True,
            use_diagonal=self.use_diagonal,
        )
        if reshaped_data.size == 0:
            logger.warning(
                "Found array with 0 sample(s) after deleting nan items"
            )
            return

        self._scaler.partial_fit(reshaped_data)
        return

    def transform(self, data: ArrayDataType) -> ArrayDataType:
        wrapped_data = phlower_array(data)
        result = wrapped_data.apply(
            self._scaler.transform,
            componentwise=self.componentwise,
            skip_nan=False,
            use_diagonal=False,
        )
        return result

    def inverse_transform(self, data: ArrayDataType) -> ArrayDataType:
        wrapped_data = phlower_array(data)
        result = wrapped_data.apply(
            self._scaler.inverse_transform,
            componentwise=self.componentwise,
            skip_nan=False,
            use_diagonal=False,
        )
        return result

    def lazy_partial_fit(self, data_files: list[IPhlowerNumpyFile]) -> None:
        for data_file in data_files:
            data = self._load_file(data_file)
            self.partial_fit(data)
        return

    def get_dumped_data(self) -> dict:
        # NOTE: Refer to setting file in order to ensure readable by program.
        _setting = ScalerInputParameters(
            method=self.method_name,
            component_wise=self.componentwise,
            parameters=vars(self._scaler),
        )
        return _setting.model_dump()

    def _load_file(
        self, numpy_file: IPhlowerNumpyFile, decrypt_key: bytes | None = None
    ) -> ArrayDataType:
        loaded_data = numpy_file.load(decrypt_key=decrypt_key)

        if numpy_file.file_extension in [
            PhlowerFileExtType.NPZENC.value,
            PhlowerFileExtType.NPZ.value,
        ]:
            if not sp.issparse(loaded_data):
                raise ValueError(
                    "Data type is not understandable for: "
                    f"{numpy_file.file_path}"
                )

        return loaded_data
