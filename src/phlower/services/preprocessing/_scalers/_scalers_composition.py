from __future__ import annotations

import pathlib

from phlower._base import IPhlowerArray
from phlower.io import PhlowerFileBuilder
from phlower.io._files import PhlowerNumpyFile
from phlower.services.preprocessing._scalers import (
    PhlowerScalerWrapper,
)
from phlower.settings import (
    PhlowerSetting,
    ScalerInputParameters,
    ScalerResolvedParameter,
)
from phlower.utils import get_logger
from phlower.utils.typing import ArrayDataType

logger = get_logger(__name__)


class ScalersComposition:
    @classmethod
    def from_setting(cls, setting: PhlowerSetting) -> ScalersComposition:
        if setting.scaling is None:
            raise ValueError(
                "scaling setting items are not found in setting file."
            )

        return cls.from_resolved_parameters(
            parameters=setting.scaling.resolve_scalers()
        )

    @classmethod
    def from_resolved_parameters(
        cls, parameters: list[ScalerResolvedParameter]
    ) -> ScalersComposition:
        scalers_dict = {
            v.scaler_name: PhlowerScalerWrapper.from_setting(v)
            for v in parameters
        }

        return cls(scalers_dict=scalers_dict)

    def __init__(
        self,
        scalers_dict: dict[str, PhlowerScalerWrapper],
    ) -> None:
        self._scalers_dict = scalers_dict

    def get_scaler_names(self) -> list[str]:
        scaler_names = list(self._scalers_dict.keys())
        return scaler_names

    def force_update(
        self, scalers_dict: dict[str, PhlowerScalerWrapper]
    ) -> None:
        self._scalers_dict |= scalers_dict

    def get_scaler(
        self, scaler_name: str, allow_missing: bool = False
    ) -> PhlowerScalerWrapper | None:
        scaler = self._scalers_dict.get(scaler_name)
        if allow_missing:
            return scaler

        if scaler is None:
            raise ValueError(f"Scaler named {scaler_name} is not found.")
        return scaler

    def transform(self, scaler_name: str, data: ArrayDataType) -> IPhlowerArray:
        scaler = self.get_scaler(scaler_name)
        transformed_data = scaler.transform(data)
        return transformed_data

    def transform_file(
        self,
        scaler_name: str,
        numpy_file: PhlowerNumpyFile | pathlib.Path,
        decrypt_key: bytes | None = None,
    ) -> IPhlowerArray:
        numpy_file = PhlowerFileBuilder.numpy_file(numpy_file)
        loaded_data = numpy_file.load(decrypt_key=decrypt_key)
        scaler = self.get_scaler(scaler_name)
        transformed_data = scaler.transform(loaded_data)
        return transformed_data

    def inverse_transform(
        self, scaler_name: str, data: ArrayDataType
    ) -> IPhlowerArray:
        scaler = self.get_scaler(scaler_name)
        return scaler.inverse_transform(data)

    def lazy_partial_fit(
        self,
        scaler_name: str,
        data_files: list[PhlowerNumpyFile],
        decrypt_key: bytes | None = None,
    ) -> None:
        scaler = self.get_scaler(scaler_name)
        scaler.lazy_partial_fit(data_files, decrypt_key=decrypt_key)
        return

    def get_dumped_data(self) -> dict[str, ScalerInputParameters]:
        dumped_data = {
            k: v.get_dumped_data() for k, v in self._scalers_dict.items()
        }
        return dumped_data
