from __future__ import annotations

import abc
import pathlib
from collections.abc import Callable
from functools import reduce
from logging import getLogger
from typing import Literal, NamedTuple, overload

import numpy as np
import pyvista as pv
from phlower_tensor import IPhlowerArray, phlower_array
from torch.utils.data import Dataset

from phlower.data._lumped_data import LumpedArrayData
from phlower.io import PhlowerDirectory
from phlower.settings import (
    ArrayDataIOSetting,
    FieldIOSettingType,
    MeshDataIOSetting,
    SlidingWindowForStage,
)
from phlower.settings._time_series_sliding_setting import (
    SlidingWindowParameters,
)
from phlower.utils._extended_simulation_field import PyVistaMeshAdapter
from phlower.utils.typing import ArrayDataType

from ._sampler import RandomPointSampler
from ._utils import BatchModeHolder

_logger = getLogger(__name__)


class IPhlowerDataset(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> LumpedArrayData: ...

    @abc.abstractmethod
    def get_members(
        self, index: int, data_type: Literal["input", "label"]
    ) -> dict[str, IPhlowerArray]:
        """Get members of the dataset for the specified data type."""
        ...

    @abc.abstractmethod
    def get_batch_mode_holder(self) -> BatchModeHolder:
        """
        Get the BatchModeFolder for the dataset, which provides information
        about the batch mode for each input, label, and field variable.
        """
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
        field_settings: list[FieldIOSettingType] | None = None,
        allow_no_y_data: bool = False,
        decrypt_key: bytes | None = None,
        random_sampler: RandomPointSampler | None = None,
        window_setting: SlidingWindowForStage | None = None,
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
            random_sampler=random_sampler,
            window_setting=window_setting,
        )

    def __init__(
        self,
        loaded_data: list[dict[str, IPhlowerArray | PyVistaMeshAdapter]],
        input_settings: list[ArrayDataIOSetting],
        label_settings: list[ArrayDataIOSetting] | None,
        *,
        field_settings: list[FieldIOSettingType] | None = None,
        allow_no_y_data: bool = False,
        decrypt_key: bytes | None = None,
        random_sampler: RandomPointSampler | None = None,
        window_setting: SlidingWindowForStage | None = None,
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
        self._random_sampler = random_sampler
        self._window_setting = window_setting
        self._sample_indices = _construct_sample_indices(
            window_setting,
            n_data=len(loaded_data),
            create_time_series_dataset=self._create_time_series_dataset,
        )

    def __len__(self):
        return len(self._sample_indices)

    def __getitem__(self, idx: int) -> LumpedArrayData:

        # NOTE: Generate random_sampler for each call to __getitem__
        # to ensure that the random sampling is independent
        # for each data item and epoch.
        if self._random_sampler is not None:
            self._random_sampler.reset()

        data_index, window_index = self._sample_indices[idx]
        x_windows, y_windows = {}, {}
        if window_index is not None:
            x_windows = _get_time_windows(
                self._window_setting.inputs, self._input_settings, window_index
            )
            y_windows = _get_time_windows(
                self._window_setting.labels, self._label_settings, window_index
            )
        x_data = self._setup_data(
            data_index,
            self._input_settings,
            allow_missing=False,
            random_sampler=self._random_sampler,
            time_windows=x_windows,
        )
        y_data = self._setup_data(
            data_index,
            self._label_settings,
            allow_missing=self._allow_no_y_data,
            random_sampler=self._random_sampler,
            time_windows=y_windows,
        )
        field_data = self._setup_data(
            data_index,
            self._field_settings,
            allow_missing=False,
            random_sampler=self._random_sampler,
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
        random_sampler: RandomPointSampler,
        time_windows: dict[str, slice] | None = None,
    ) -> dict[str, IPhlowerArray | PyVistaMeshAdapter]:
        time_windows = time_windows or {}
        _dict = {
            setting.name: _apply_setting(
                dict_arrs=self._loaded_data[index],
                io_setting=setting,
                allow_missing=allow_missing,
                random_sampler=random_sampler,
                time_window=time_windows.get(setting.name),
            )
            for setting in variable_settings
        }
        return {name: v for name, v in _dict.items() if v is not None}

    def _create_time_series_dataset(self) -> OnMemoryPhlowerDataSet:
        return OnMemoryPhlowerDataSet(
            loaded_data=self._loaded_data,
            input_settings=_time_series_settings(self._input_settings),
            label_settings=_time_series_settings(self._label_settings),
            allow_no_y_data=self._allow_no_y_data,
        )

    def get_batch_mode_holder(self) -> BatchModeHolder:
        return _get_batch_mode_holder(self)


# endregion

# region LazyPhlowerDataset


class LazyPhlowerDataset(Dataset, IPhlowerDataset):
    """This Dataset reads input file every time to get data"""

    @classmethod
    def create(
        cls,
        input_settings: list[ArrayDataIOSetting],
        label_settings: list[ArrayDataIOSetting],
        directories: list[pathlib.Path],
        *,
        field_settings: list[FieldIOSettingType] | None = None,
        allow_no_y_data: bool = False,
        decrypt_key: bytes | None = None,
        random_sampler: RandomPointSampler | None = None,
        window_setting: SlidingWindowForStage | None = None,
    ) -> LazyPhlowerDataset:
        return LazyPhlowerDataset(
            input_settings=input_settings,
            label_settings=label_settings,
            directories=directories,
            field_settings=field_settings,
            allow_no_y_data=allow_no_y_data,
            decrypt_key=decrypt_key,
            random_sampler=random_sampler,
            window_setting=window_setting,
        )

    def __init__(
        self,
        input_settings: list[ArrayDataIOSetting],
        label_settings: list[ArrayDataIOSetting] | None,
        directories: list[pathlib.Path],
        *,
        field_settings: list[FieldIOSettingType] | None = None,
        allow_no_y_data: bool = False,
        decrypt_key: bytes | None = None,
        random_sampler: RandomPointSampler | None = None,
        window_setting: SlidingWindowForStage | None = None,
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
        self._random_sampler = random_sampler
        self._window_setting = window_setting
        self._sample_indices = _construct_sample_indices(
            window_setting,
            n_data=len(self._directories),
            create_time_series_dataset=self._create_time_series_dataset,
        )

    def __len__(self):
        return len(self._sample_indices)

    def __getitem__(self, idx: int) -> LumpedArrayData:

        if self._random_sampler is not None:
            self._random_sampler.reset()

        data_index, window_index = self._sample_indices[idx]
        data_directory = self._directories[data_index]
        x_windows, y_windows = {}, {}
        if window_index is not None:
            x_windows = _get_time_windows(
                self._window_setting.inputs, self._input_settings, window_index
            )
            y_windows = _get_time_windows(
                self._window_setting.labels, self._label_settings, window_index
            )
        x_data = self._setup_data(
            data_directory,
            self._input_settings,
            allow_missing=False,
            random_sampler=self._random_sampler,
            time_windows=x_windows,
        )
        y_data = self._setup_data(
            data_directory,
            self._label_settings,
            allow_missing=self._allow_no_y_data,
            random_sampler=self._random_sampler,
            time_windows=y_windows,
        )
        field_data = self._setup_data(
            data_directory,
            self._field_settings,
            allow_missing=False,
            random_sampler=self._random_sampler,
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
        random_sampler: RandomPointSampler,
        time_windows: dict[str, slice] | None = None,
    ) -> dict[str, IPhlowerArray | PyVistaMeshAdapter]:

        return _load_data(
            data_directory=data_directory,
            settings=variable_settings,
            allow_missing=allow_missing,
            decrypt_key=self._decrypt_key,
            random_sampler=random_sampler,
            time_windows=time_windows,
        )

    def _create_time_series_dataset(self) -> LazyPhlowerDataset:
        return LazyPhlowerDataset(
            input_settings=_time_series_settings(self._input_settings),
            label_settings=_time_series_settings(self._label_settings),
            directories=[d.path for d in self._directories],
            allow_no_y_data=self._allow_no_y_data,
            decrypt_key=self._decrypt_key,
        )

    def get_batch_mode_holder(self) -> BatchModeHolder:
        return _get_batch_mode_holder(self)


# endregion

# region utility functions for loading data


@overload
def _load_data(
    data_directory: PhlowerDirectory,
    settings: list[ArrayDataIOSetting],
    allow_missing: bool,
    decrypt_key: bytes | None = None,
    skip_apply_setting: Literal[False] | None = ...,
    random_sampler: RandomPointSampler | None = None,
    time_windows: dict[str, slice] | None = None,
) -> dict[str, IPhlowerArray | PyVistaMeshAdapter]: ...


@overload
def _load_data(
    data_directory: PhlowerDirectory,
    settings: list[ArrayDataIOSetting],
    allow_missing: bool,
    decrypt_key: bytes | None = None,
    skip_apply_setting: Literal[True] = True,
    random_sampler: RandomPointSampler | None = None,
    time_windows: dict[str, slice] | None = None,
) -> dict[str, np.ndarray | pv.DataSet]: ...


def _load_data(
    data_directory: PhlowerDirectory,
    settings: list[FieldIOSettingType],
    allow_missing: bool,
    decrypt_key: bytes | None = None,
    skip_apply_setting: bool = False,
    random_sampler: RandomPointSampler | None = None,
    time_windows: dict[str, slice] | None = None,
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

    time_windows = time_windows or {}
    _collected = {
        setting.name: _apply_setting(
            _results,
            setting,
            allow_missing=allow_missing,
            random_sampler=random_sampler,
            time_window=time_windows.get(setting.name),
        )
        for setting in settings
    }
    return {name: arr for name, arr in _collected.items() if arr is not None}


def _load_ndarray_array(
    data_directory: PhlowerDirectory,
    io_setting: FieldIOSettingType,
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
    io_setting: FieldIOSettingType,
    allow_missing: bool = False,
    random_sampler: RandomPointSampler | None = None,
    time_window: slice | None = None,
) -> IPhlowerArray | PyVistaMeshAdapter | None:

    match io_setting:
        case ArrayDataIOSetting():
            return _apply_setting_to_array(
                dict_arrs,
                io_setting,
                allow_missing=allow_missing,
                random_sampler=random_sampler,
                time_window=time_window,
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
    io_setting: FieldIOSettingType,
    allow_missing: bool = False,
    random_sampler: RandomPointSampler | None = None,
    time_window: slice | None = None,
) -> IPhlowerArray | pv.UnstructuredGrid | None:
    member_arrs: list[np.ndarray] = _extract_member_arrays(
        io_setting, dict_arrs
    )

    if len(member_arrs) == 0:
        if allow_missing:
            return None
        raise ValueError(f"Arrays for {io_setting.name} are missing.")

    # Slice in time before concatenating and casting,
    # which copy the arrays.
    member_arrs = [
        _slice_time(arr, io_setting, time_window) for arr in member_arrs
    ]

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
        dtype=io_setting.dtype,
    )
    # -- random sampling
    if io_setting.random_sampling.is_active:
        if random_sampler is None:
            _logger.info(
                f"Random sampling is active for {io_setting.name}, "
                "but no RandomSampler is provided. "
                "Random sampling will not be applied."
            )
        else:
            _array = random_sampler.sample(io_setting, _array)

    return _array


def _extract_member_arrays(
    io_setting: ArrayDataIOSetting, dict_arrs: dict[str, np.ndarray]
) -> list[np.ndarray]:
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
                f"setting: {member.n_last_dim}, actual: {_arr.shape}, "
                f"{member=}"
            )

        if member.index_first_dim is not None:
            _arr = _arr[member.index_first_dim]
        member_arrs.append(_arr)
    return member_arrs


def _slice_time(
    arr: np.ndarray,
    io_setting: ArrayDataIOSetting,
    time_window: slice | None,
) -> np.ndarray:
    if io_setting.time_slice is None and time_window is None:
        return arr

    if not isinstance(arr, np.ndarray):
        raise ValueError(
            f"Time slicing of {io_setting.name} needs a dense array."
        )
    if io_setting.time_slice is not None:
        arr = arr[io_setting.time_slice_object]
    if time_window is not None:
        arr = arr[time_window]
    return arr


# endregion


# region windows as samples


class SampleIndex(NamedTuple):
    data_index: int
    window_index: int | None


class SampleIndices:
    """Data index and window index of every sample of a dataset.

    Only the cumulative number of windows of each data is stored,
    so that no list of every sample is created.
    """

    def __init__(
        self, n_data: int, n_windows_per_data: np.ndarray | None = None
    ):
        self._cumulative_n_windows = (
            None
            if n_windows_per_data is None
            else np.cumsum(n_windows_per_data)
        )
        self._n_samples = (
            n_data
            if n_windows_per_data is None
            else int(np.sum(n_windows_per_data))
        )

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, index: int) -> SampleIndex:
        if not -self._n_samples <= index < self._n_samples:
            raise IndexError(
                f"Index {index} is out of range for {self._n_samples} samples."
            )
        if index < 0:
            index += self._n_samples
        if self._cumulative_n_windows is None:
            return SampleIndex(data_index=index, window_index=None)

        data_index = int(
            np.searchsorted(self._cumulative_n_windows, index, side="right")
        )
        first_sample_index = (
            0
            if data_index == 0
            else int(self._cumulative_n_windows[data_index - 1])
        )
        return SampleIndex(
            data_index=data_index, window_index=index - first_sample_index
        )


def _construct_sample_indices(
    window_setting: SlidingWindowForStage | None,
    n_data: int,
    create_time_series_dataset: Callable[[], IPhlowerDataset],
) -> SampleIndices:
    """Indices of the samples of a dataset.

    When windows are samples, every window of every data is a sample.
    Otherwise, every data is a sample and the window index is None.
    """
    if window_setting is None or not window_setting.windows_as_samples():
        return SampleIndices(n_data)

    time_series_dataset = create_time_series_dataset()
    n_windows_per_data = np.empty(n_data, dtype=np.int64)
    for i in range(n_data):
        sample = time_series_dataset[i]
        counts = window_setting.inputs.get_unique_window_counts(
            {name: array.shape[0] for name, array in sample.x_data.items()}
        ) | window_setting.labels.get_unique_window_counts(
            {name: array.shape[0] for name, array in sample.y_data.items()}
        )
        if len(counts) == 0:
            raise ValueError(
                "treat_as_samples needs time series inputs or labels."
            )
        if len(counts) > 1:
            raise ValueError(
                "The number of windows must be the same for all time series "
                f"of data {i}. Found: {sorted(counts)}"
            )
        n_windows = counts.pop()
        if n_windows == 0:
            raise ValueError(
                f"No window is found in data {i}. "
                "The time series is shorter than the window."
            )
        n_windows_per_data[i] = n_windows
    return SampleIndices(n_data, n_windows_per_data)


def _time_series_settings(
    io_settings: list[ArrayDataIOSetting],
) -> list[ArrayDataIOSetting]:
    return [setting for setting in io_settings if setting.is_time_series]


def _get_time_windows(
    window_parameters: SlidingWindowParameters,
    io_settings: list[ArrayDataIOSetting],
    window_index: int,
) -> dict[str, slice]:
    """Slice of the window of each time series variable."""
    return {
        setting.name: window_parameters.get_slice(setting.name, window_index)
        for setting in io_settings
        if setting.is_time_series
    }


# endregion

# region utility functions for batch mode


def _get_batch_mode_holder(
    self: OnMemoryPhlowerDataSet | LazyPhlowerDataset,
) -> BatchModeHolder:
    inputs_batch_mode = {
        setting.name: setting.batch_mode for setting in self._input_settings
    }
    labels_batch_mode = {
        setting.name: setting.batch_mode for setting in self._label_settings
    }
    field_batch_mode = {
        setting.name: setting.batch_mode for setting in self._field_settings
    }
    return BatchModeHolder(
        inputs_batch_mode=inputs_batch_mode,
        labels_batch_mode=labels_batch_mode,
        field_batch_mode=field_batch_mode,
    )


# endregion
