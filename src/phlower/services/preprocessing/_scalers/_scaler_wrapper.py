from __future__ import annotations

from collections.abc import Callable

import numpy as np
from phlower_tensor import IPhlowerArray, phlower_array

from phlower.io._files import IPhlowerNumpyFile
from phlower.services.preprocessing._scalers import (
    scale_functions,
)
from phlower.settings import ScalerInputParameters, ScalerResolvedParameter
from phlower.utils import get_logger
from phlower.utils.enums import PhlowerFileExtType
from phlower.utils.typing import ArrayDataType

logger = get_logger(__name__)


class PhlowerScalerWrapper:
    @classmethod
    def from_setting(
        cls, setting: ScalerResolvedParameter
    ) -> PhlowerScalerWrapper:
        _cls = cls(
            method=setting.method,
            componentwise=setting.component_wise,
            component_axis=setting.component_axis,
            parameters=setting.parameters,
        )
        return _cls

    def __init__(
        self,
        method: str,
        *,
        componentwise: bool = False,
        component_axis: int = -1,
        parameters: dict | None = None,
    ):
        if parameters is None:
            parameters = {}
        self.method_name = method
        self._scaler = scale_functions.create_scaler(method, **parameters)
        self.componentwise = componentwise
        self.component_axis = component_axis

    @property
    def use_diagonal(self) -> bool:
        return self._scaler.use_diagonal

    def is_erroneous(self) -> bool:
        return self._scaler.is_erroneous()

    def is_fitted(self) -> bool:
        return self._scaler.is_fitted()

    def partial_fit(self, data: ArrayDataType) -> None:
        wrapped_data = self._move_component_axis(
            phlower_array(data), to_last=True
        )
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

        self._scaler.partial_fit(reshaped_data.to_numpy())
        return

    def transform(self, data: ArrayDataType) -> IPhlowerArray:
        return self._apply(self._scaler.transform, data)

    def inverse_transform(self, data: ArrayDataType) -> ArrayDataType:
        return self._apply(self._scaler.inverse_transform, data)

    def _apply(
        self,
        function: Callable[[ArrayDataType], ArrayDataType],
        data: ArrayDataType,
    ) -> IPhlowerArray:
        wrapped_data = self._move_component_axis(
            phlower_array(data), to_last=True
        )
        result = wrapped_data.apply(
            function,
            componentwise=self.componentwise,
            skip_nan=False,
            use_diagonal=False,
        )
        return self._move_component_axis(result, to_last=False)

    def _move_component_axis(
        self, data: IPhlowerArray, to_last: bool
    ) -> IPhlowerArray:
        """Move the component axis to the last axis, where the scaler
        sees the components, or back to its place."""
        if self.component_axis == -1:
            return data
        if len(data.shape) < 1 - self.component_axis:
            raise ValueError(
                f"Data of shape {data.shape} has no axis {self.component_axis} "
                "of components."
            )

        source, destination = (self.component_axis, -1)
        if not to_last:
            source, destination = destination, source
        return phlower_array(
            np.moveaxis(data.to_numpy(), source, destination),
            is_time_series=data.is_time_series,
            is_voxel=data.is_voxel,
            dimensions=data.dimension,
        )

    def lazy_partial_fit(
        self,
        data_files: list[IPhlowerNumpyFile],
        decrypt_key: bytes | None = None,
    ) -> None:
        for data_file in data_files:
            data = self._load_file(data_file, decrypt_key=decrypt_key)
            self.partial_fit(data)
        return

    def get_dumped_data(self) -> ScalerInputParameters:
        # NOTE: Refer to setting file in order to ensure readable by program.
        _setting = ScalerInputParameters(
            method=self.method_name,
            component_wise=self.componentwise,
            component_axis=self.component_axis,
            parameters=self._scaler.get_dumped_data(),
        )
        return _setting

    def _load_file(
        self, numpy_file: IPhlowerNumpyFile, decrypt_key: bytes | None = None
    ) -> ArrayDataType:
        loaded_data = numpy_file.load(decrypt_key=decrypt_key)

        if numpy_file.file_extension in [
            PhlowerFileExtType.NPZENC.value,
            PhlowerFileExtType.NPZ.value,
        ]:
            if not loaded_data.is_sparse:
                raise ValueError(
                    "Data type is not understandable for: "
                    f"{numpy_file.file_path}"
                )

        return loaded_data
