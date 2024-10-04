import abc
import pathlib

import numpy as np
from torch.utils.data import Dataset

from phlower._base.array import IPhlowerArray, phlower_array
from phlower.data._lumped_data import LumpedArrayData
from phlower.io import PhlowerDirectory
from phlower.settings import ModelIOSetting
from phlower.settings._model_setting import _MemberSetting


class IPhlowerDataset(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> LumpedArrayData: ...


class LazyPhlowerDataset(Dataset, IPhlowerDataset):
    """This Dataset reads input file every time to get data"""

    def __init__(
        self,
        input_settings: list[ModelIOSetting],
        label_settings: list[ModelIOSetting] | None,
        directories: list[pathlib.Path],
        *,
        field_settings: list[ModelIOSetting] | None = None,
        allow_no_y_data: bool = False,
        decrypt_key: bytes | None = None,
    ):
        self._input_settings = input_settings
        self._label_settings = (
            label_settings if label_settings is not None else []
        )
        self._directories = [PhlowerDirectory(d) for d in directories]
        self._field_settings = (
            field_settings if field_settings is not None else []
        )

        self._allow_no_y_data = allow_no_y_data
        self._decrypt_key = decrypt_key

    def __len__(self):
        return len(self._directories)

    def __getitem__(self, idx: int) -> LumpedArrayData:
        data_directory = self._directories[idx]
        x_data = self._setup_data(
            data_directory, self._input_settings, allow_missing=False
        )
        y_data = self._setup_data(
            data_directory,
            self._label_settings,
            allow_missing=self._allow_no_y_data,
        )
        field_data = self._setup_data(
            data_directory, self._field_settings, allow_missing=False
        )
        return LumpedArrayData(
            x_data=x_data,
            y_data=y_data,
            field_data=field_data,
            data_directory=data_directory,
        )

    def _setup_data(
        self,
        data_directory: PhlowerDirectory,
        variable_settings: list[ModelIOSetting],
        allow_missing: bool,
    ) -> dict[str, IPhlowerArray]:
        arrs = [
            (
                setting.name,
                self._load_phlower_array(
                    data_directory, setting, allow_missing=allow_missing
                ),
            )
            for setting in variable_settings
        ]

        return {name: arr for name, arr in arrs if arr is not None}

    def _load_phlower_array(
        self,
        data_directory: PhlowerDirectory,
        io_setting: ModelIOSetting,
        allow_missing: bool,
    ) -> IPhlowerArray | None:
        arr = self._load_ndarray_data(
            data_directory=data_directory,
            members=io_setting.members,
            allow_missing=allow_missing,
        )

        if arr is None:
            return None

        return phlower_array(
            arr,
            is_time_series=io_setting.is_time_series,
            is_voxel=io_setting.is_voxel,
            dimensions=io_setting.physical_dimension,
        )

    def _load_ndarray_data(
        self,
        data_directory: PhlowerDirectory,
        members: list[_MemberSetting],
        allow_missing: bool,
    ) -> np.ndarray | None:
        arrs: list[np.ndarray] = []

        for member in members:
            file_path = data_directory.find_variable_file(
                variable_name=member.name, allow_missing=allow_missing
            )
            if file_path is None:
                continue

            _arr = file_path.load(decrypt_key=self._decrypt_key).to_numpy()

            if member.n_last_dim is None:
                arrs.append(_arr)
                continue

            if _arr.shape[-1] == member.n_last_dim:
                arrs.append(_arr)
                continue

            if member.n_last_dim == 1:
                _arr = _arr[:, np.newaxis]
                arrs.append(_arr)
                continue

            raise ValueError(
                "Last dimension does not match. "
                f"setting: {member.n_last_dim}, actual: {_arr.shape[-1]}"
            )

        if len(arrs) == 0:
            return None

        if len(arrs) == 1:
            return arrs[0]

        return np.concatenate(arrs)
