from __future__ import annotations

import abc
import pathlib
from functools import reduce
from typing import Literal, overload

import numpy as np
import pyvista as pv
from phlower_tensor import IPhlowerArray, phlower_array
from torch.utils.data import Dataset

from phlower.data._lumped_data import LumpedArrayData
from phlower.io import PhlowerDirectory
from phlower.settings import (
    ArrayDataIOSetting,
    MeshDataIOSetting,
    ModelIOSettingType,
)
from phlower.utils._extended_simulation_field import PyVistaMeshAdapter
from phlower.utils.typing import ArrayDataType


class IPhlowerDataset(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> LumpedArrayData: ...

    @abc.abstractmethod
    def get_members(
        self, index: int, data_type: Literal["input", "label"]
    ) -> dict[str, IPhlowerArray]:
        """Get members of the dataset for the specified data type."""
        ...


# region OnMemoryPhlowerDataset


class OnMemoryPhlowerDataSet(Dataset, IPhlowerDataset):
    @classmethod
    def create(
        cls,
        input_settings: list[ArrayDataIOSetting],
        label_settings: list[ArrayDataIOSetting],
        directories: list[pathlib.Path],
        *,
        field_settings: list[ModelIOSettingType] | None = None,
        allow_no_y_data: bool = False,
        decrypt_key: bytes | None = None,
    ) -> OnMemoryPhlowerDataSet:
        field_settings = field_settings or []

        _all_settings = input_settings + label_settings + field_settings
        all_loaded_data = [
            _load_data(
                data_directory=PhlowerDirectory(directory),
                settings=_all_settings,
                allow_missing=True,
                decrypt_key=decrypt_key,
                skip_apply_setting=True,
            )
            for directory in directories
        ]
        return OnMemoryPhlowerDataSet(
            loaded_data=all_loaded_data,
            input_settings=input_settings,
            label_settings=label_settings,
            field_settings=field_settings,
            allow_no_y_data=allow_no_y_data,
            decrypt_key=decrypt_key,
        )

    def __init__(
        self,
        loaded_data: list[dict[str, IPhlowerArray | PyVistaMeshAdapter]],
        input_settings: list[ArrayDataIOSetting],
        label_settings: list[ArrayDataIOSetting] | None,
        *,
        field_settings: list[ModelIOSettingType] | None = None,
        allow_no_y_data: bool = False,
        decrypt_key: bytes | None = None,
    ):
        self._loaded_data = loaded_data
        self._input_settings = input_settings
        self._label_settings = (
            label_settings if label_settings is not None else []
        )
        self._field_settings = (
            field_settings if field_settings is not None else []
        )

        self._allow_no_y_data = allow_no_y_data
        self._decrypt_key = decrypt_key

    def __len__(self):
        return len(self._loaded_data)

    def __getitem__(self, idx: int) -> LumpedArrayData:
        x_data = self._setup_data(
            idx, self._input_settings, allow_missing=False
        )
        y_data = self._setup_data(
            idx,
            self._label_settings,
            allow_missing=self._allow_no_y_data,
        )
        field_data = self._setup_data(
            idx, self._field_settings, allow_missing=False
        )
        return LumpedArrayData(
            x_data=x_data, y_data=y_data, field_data=field_data
        )

    def get_members(
        self, index: int, data_type: Literal["input", "label"]
    ) -> dict[str, IPhlowerArray]:
        match data_type:
            case "input":
                target_settings = self._input_settings
            case "label":
                target_settings = self._label_settings
            case _:
                raise ValueError(
                    f"Invalid data type: {data_type}. "
                    "Must be 'input' or 'label'."
                )

        return {
            name: phlower_array(arr)
            for name, arr in self._loaded_data[index].items()
            if any(setting.has_member(name) for setting in target_settings)
        }

    def _setup_data(
        self,
        index: int,
        variable_settings: list[ArrayDataIOSetting],
        allow_missing: bool,
    ) -> dict[str, IPhlowerArray | PyVistaMeshAdapter]:
        _dict = {
            setting.name: _apply_setting(
                dict_arrs=self._loaded_data[index],
                io_setting=setting,
                allow_missing=allow_missing,
            )
            for setting in variable_settings
        }
        return {name: v for name, v in _dict.items() if v is not None}


# endregion

# region LazyPhlowerDataset


class LazyPhlowerDataset(Dataset, IPhlowerDataset):
    """This Dataset reads input file every time to get data"""

    def __init__(
        self,
        input_settings: list[ArrayDataIOSetting],
        label_settings: list[ArrayDataIOSetting] | None,
        directories: list[pathlib.Path],
        *,
        field_settings: list[ModelIOSettingType] | None = None,
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

    def get_members(
        self, index: int, data_type: Literal["input", "label"]
    ) -> dict[str, IPhlowerArray]:
        match data_type:
            case "input":
                io_settings = self._input_settings
            case "label":
                io_settings = self._label_settings
            case _:
                raise ValueError(
                    f"Invalid data type: {data_type}. "
                    "Must be 'input' or 'label'."
                )

        arrays = [
            _load_ndarray_array(
                data_directory=self._directories[index],
                io_setting=setting,
                allow_missing=True,
                decrypt_key=self._decrypt_key,
            )
            for setting in io_settings
        ]
        arrays = reduce(lambda x, y: x | y, arrays, {})
        # NOTE: Just convert phlower_array without any information
        return {name: phlower_array(arr) for name, arr in arrays.items()}

    def _setup_data(
        self,
        data_directory: PhlowerDirectory,
        variable_settings: list[ArrayDataIOSetting],
        allow_missing: bool,
    ) -> dict[str, IPhlowerArray | PyVistaMeshAdapter]:

        return _load_data(
            data_directory=data_directory,
            settings=variable_settings,
            allow_missing=allow_missing,
            decrypt_key=self._decrypt_key,
        )


# endregion

# region utility functions for loading data


@overload
def _load_data(
    data_directory: PhlowerDirectory,
    settings: list[ArrayDataIOSetting],
    allow_missing: bool,
    decrypt_key: bytes | None = None,
    skip_apply_setting: Literal[False] | None = ...,
) -> dict[str, IPhlowerArray | PyVistaMeshAdapter]: ...


@overload
def _load_data(
    data_directory: PhlowerDirectory,
    settings: list[ArrayDataIOSetting],
    allow_missing: bool,
    decrypt_key: bytes | None = None,
    skip_apply_setting: Literal[True] = True,
) -> dict[str, np.ndarray | pv.DataSet]: ...


def _load_data(
    data_directory: PhlowerDirectory,
    settings: list[ModelIOSettingType],
    allow_missing: bool,
    decrypt_key: bytes | None = None,
    skip_apply_setting: bool = False,
) -> (
    dict[str, IPhlowerArray | PyVistaMeshAdapter]
    | dict[str, np.ndarray | pv.DataSet]
):

    _results = {}
    for setting in settings:
        _out = None
        match setting:
            case ArrayDataIOSetting():
                _out = _load_ndarray_array(
                    data_directory,
                    setting,
                    allow_missing=allow_missing,
                    decrypt_key=decrypt_key,
                )
            case MeshDataIOSetting():
                _out = _load_mesh_data(
                    data_directory,
                    setting,
                    allow_missing=allow_missing,
                    decrypt_key=decrypt_key,
                )
            case _:
                raise ValueError(f"Unknown setting type: {type(setting)}")

        _results.update(_out)

    if skip_apply_setting:
        return _results

    _collected = {
        setting.name: _apply_setting(
            _results, setting, allow_missing=allow_missing
        )
        for setting in settings
    }
    return {name: arr for name, arr in _collected.items() if arr is not None}


def _load_ndarray_array(
    data_directory: PhlowerDirectory,
    io_setting: ModelIOSettingType,
    allow_missing: bool,
    decrypt_key: bytes | None = None,
) -> dict[str, np.ndarray]:

    assert isinstance(io_setting, ArrayDataIOSetting)

    name_to_arr: dict[str, np.ndarray] = {}
    for member in io_setting.members:
        ph_path = data_directory.find_variable_file(
            variable_name=member.name, allow_missing=allow_missing
        )
        if ph_path is None:
            continue

        _arr = ph_path.load(decrypt_key=decrypt_key).to_numpy()
        name_to_arr[ph_path.get_variable_name()] = _arr

    return name_to_arr


def _load_mesh_data(
    data_directory: PhlowerDirectory,
    setting: MeshDataIOSetting,
    allow_missing: bool,
    decrypt_key: bytes | None = None,
) -> dict[str, pv.DataSet]:

    target_path = data_directory.path / setting.filename
    if not target_path.exists():
        if allow_missing:
            return {}
        raise ValueError(f"Mesh data file not found: {target_path}")

    mesh = pv.read(target_path)
    return {setting.name: mesh}


def _apply_setting(
    dict_arrs: dict[str, np.ndarray | IPhlowerArray | pv.PointGrid],
    io_setting: ModelIOSettingType,
    allow_missing: bool = False,
) -> IPhlowerArray | PyVistaMeshAdapter | None:

    match io_setting:
        case ArrayDataIOSetting():
            return _apply_setting_to_array(
                dict_arrs, io_setting, allow_missing=allow_missing
            )
        case MeshDataIOSetting():
            return _apply_setting_to_mesh(
                dict_arrs, io_setting, allow_missing=allow_missing
            )
        case _:
            raise ValueError(f"Unknown IO setting type: {type(io_setting)}")


def _apply_setting_to_mesh(
    dict_arrs: dict[str, np.ndarray | IPhlowerArray | pv.PointGrid],
    io_setting: MeshDataIOSetting,
    allow_missing: bool = False,
) -> PyVistaMeshAdapter | None:
    if io_setting.name not in dict_arrs:
        if allow_missing:
            return None
        raise ValueError(f"Mesh data for {io_setting.name} is missing.")

    mesh = dict_arrs[io_setting.name]
    if not isinstance(mesh, pv.DataSet):
        raise ValueError(
            f"Data for {io_setting.name} must be a PyVista DataSet."
        )

    return PyVistaMeshAdapter(
        mesh, dimensions=io_setting.physical_dimension_collection
    )


def _apply_setting_to_array(
    dict_arrs: dict[str, np.ndarray | IPhlowerArray | pv.PointGrid],
    io_setting: ModelIOSettingType,
    allow_missing: bool = False,
) -> IPhlowerArray | pv.UnstructuredGrid | None:
    member_arrs: list[np.ndarray] = []
    for member in io_setting.members:
        if member.name not in dict_arrs:
            continue

        _arr = dict_arrs[member.name]
        if not isinstance(_arr, (ArrayDataType, IPhlowerArray)):
            raise ValueError(
                f"Data for {member.name} must be "
                "either numpy array or PhlowerArray."
            )

        if isinstance(_arr, IPhlowerArray):
            _arr = _arr.to_numpy()

        if member.n_last_dim is None:
            pass
        elif _arr.shape[-1] == member.n_last_dim:
            pass
        elif member.n_last_dim == 1:
            _arr = _arr[..., np.newaxis]
        else:
            raise ValueError(
                "Last dimension does not match. "
                f"setting: {member.n_last_dim}, actual: {_arr.shape[-1]}"
            )

        if member.index_first_dim is not None:
            _arr = _arr[member.index_first_dim]
        member_arrs.append(_arr)

    if len(member_arrs) == 0:
        if allow_missing:
            return None
        raise ValueError(f"Arrays for {io_setting.name} are missing.")

    _array = (
        member_arrs[0]
        if len(member_arrs) == 1
        else np.concatenate(member_arrs, axis=-1)
    )
    _array = phlower_array(
        _array,
        is_time_series=io_setting.is_time_series,
        is_voxel=io_setting.is_voxel,
        dimensions=io_setting.physical_dimension,
    )
    if io_setting.time_slice is None:
        return _array

    return _array.slice_along_time_axis(io_setting.time_slice_object)


# endregion
