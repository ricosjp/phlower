import pathlib

import numpy as np
import pytest
import scipy.sparse as sp

from phlower.data import LazyPhlowerDataset, OnMemoryPhlowerDataSet
from phlower.data._datasets import SampleIndices, _slice_time
from phlower.io import PhlowerNumpyFile
from phlower.settings import ArrayDataIOSetting, SlidingWindowForStage
from phlower.utils.sliding_window import SlidingWindow

_N_NODES = 4
_DATASET_CLASSES = [OnMemoryPhlowerDataSet, LazyPhlowerDataset]


def _create_time_series(
    output_directory: pathlib.Path,
    time_lengths: list[int],
    variable_names: list[str] | None = None,
) -> list[pathlib.Path]:
    directories = []
    for i, n_time in enumerate(time_lengths):
        directory = output_directory / f"data{i}"
        directory.mkdir(parents=True)
        for name in variable_names or ["u"]:
            PhlowerNumpyFile.save(
                directory, name, np.random.rand(n_time, _N_NODES, 2)
            )
        PhlowerNumpyFile.save(directory, "x", np.random.rand(_N_NODES, 3))
        directories.append(directory)
    return directories


def _create_dataset(
    dataset_cls: type[OnMemoryPhlowerDataSet | LazyPhlowerDataset],
    directories: list[pathlib.Path],
    window_setting: SlidingWindowForStage | None,
    allow_no_y_data: bool = False,
) -> OnMemoryPhlowerDataSet | LazyPhlowerDataset:
    # Label is the next time step of the input
    input_settings = [
        ArrayDataIOSetting(
            name="u",
            members=[{"name": "u"}],
            is_time_series=True,
            time_slice=[0, -1],
        ),
        ArrayDataIOSetting(name="x", members=[{"name": "x"}]),
    ]
    label_settings = [
        ArrayDataIOSetting(
            name="u_next",
            members=[{"name": "u"}],
            is_time_series=True,
            time_slice=[1, None],
        ),
    ]
    return dataset_cls.create(
        input_settings=input_settings,
        label_settings=label_settings,
        directories=directories,
        window_setting=window_setting,
        allow_no_y_data=allow_no_y_data,
    )


@pytest.mark.parametrize("dataset_cls", _DATASET_CLASSES)
@pytest.mark.parametrize(
    "offset, size, stride", [(0, 1, 1), (0, 2, 1), (1, 2, 2)]
)
def test__windows_as_samples(
    dataset_cls: type[OnMemoryPhlowerDataSet | LazyPhlowerDataset],
    offset: int,
    size: int,
    stride: int,
    tmp_path: pathlib.Path,
):
    time_lengths = [5, 8]
    directories = _create_time_series(tmp_path, time_lengths)
    window = SlidingWindow(offset=offset, size=size, stride=stride)
    dataset = _create_dataset(
        dataset_cls,
        directories,
        SlidingWindowForStage(
            inputs=window, labels=window, treat_as_samples=True
        ),
    )

    # time_slice removes one time step from the inputs and labels.
    n_windows = [window.get_total_items(n - 1) for n in time_lengths]
    assert len(dataset) == sum(n_windows)

    items = [(i, j) for i, n in enumerate(n_windows) for j in range(n)]
    for idx, (i, j) in enumerate(items):
        u = PhlowerNumpyFile(directories[i] / "u.npy").load().to_numpy()
        x = PhlowerNumpyFile(directories[i] / "x.npy").load().to_numpy()
        item = dataset[idx]

        np.testing.assert_array_almost_equal(
            item.x_data["u"].to_numpy(), u[:-1][window.get_slice(j)]
        )
        np.testing.assert_array_almost_equal(
            item.y_data["u_next"].to_numpy(), u[1:][window.get_slice(j)]
        )
        np.testing.assert_array_almost_equal(item.x_data["x"].to_numpy(), x)


@pytest.mark.parametrize("dataset_cls", _DATASET_CLASSES)
def test__windows_as_samples_with_different_windows(
    dataset_cls: type[OnMemoryPhlowerDataSet | LazyPhlowerDataset],
    tmp_path: pathlib.Path,
):
    # Two input steps predict the following step.
    n_time = 6
    directories = _create_time_series(tmp_path, [n_time], ["u", "v"])
    input_settings = [
        ArrayDataIOSetting(
            name="uv",
            members=[{"name": "u"}, {"name": "v"}],
            is_time_series=True,
            time_slice=[0, -1],
        ),
    ]
    label_settings = [
        ArrayDataIOSetting(
            name="u", members=[{"name": "u"}], is_time_series=True
        ),
    ]
    window_setting = SlidingWindowForStage(
        inputs={"uv": {"offset": 0, "size": 2, "stride": 1}},
        labels={"offset": 2, "size": 1, "stride": 1},
        treat_as_samples=True,
    )
    dataset = dataset_cls.create(
        input_settings=input_settings,
        label_settings=label_settings,
        directories=directories,
        window_setting=window_setting,
    )

    assert len(dataset) == n_time - 2
    u = PhlowerNumpyFile(directories[0] / "u.npy").load().to_numpy()
    v = PhlowerNumpyFile(directories[0] / "v.npy").load().to_numpy()
    for j in range(n_time - 2):
        item = dataset[j]
        np.testing.assert_array_almost_equal(
            item.x_data["uv"].to_numpy(),
            np.concatenate([u, v], axis=-1)[j : j + 2],
        )
        np.testing.assert_array_almost_equal(
            item.y_data["u"].to_numpy(), u[j + 2 : j + 3]
        )


@pytest.mark.parametrize(
    "window_setting",
    [
        None,
        SlidingWindowForStage(is_active=False),
        SlidingWindowForStage(is_active=True),
        SlidingWindowForStage(is_active=False, treat_as_samples=True),
    ],
)
@pytest.mark.parametrize("dataset_cls", _DATASET_CLASSES)
def test__one_data_is_one_sample_when_windows_are_not_samples(
    window_setting: SlidingWindowForStage | None,
    dataset_cls: type[OnMemoryPhlowerDataSet | LazyPhlowerDataset],
    tmp_path: pathlib.Path,
):
    time_lengths = [5, 7]
    directories = _create_time_series(tmp_path, time_lengths)
    dataset = _create_dataset(dataset_cls, directories, window_setting)

    assert len(dataset) == len(time_lengths)
    for i, n_time in enumerate(time_lengths):
        assert dataset[i].x_data["u"].shape == (n_time - 1, _N_NODES, 2)


@pytest.mark.parametrize("dataset_cls", _DATASET_CLASSES)
def test__skip_missing_labels_when_allowed(
    dataset_cls: type[OnMemoryPhlowerDataSet | LazyPhlowerDataset],
    tmp_path: pathlib.Path,
):
    directories = _create_time_series(tmp_path, [5], ["u", "v"])
    window = {"offset": 0, "size": 1, "stride": 1}
    window_setting = SlidingWindowForStage(
        inputs=window, labels=window, treat_as_samples=True
    )
    dataset = dataset_cls.create(
        input_settings=[
            ArrayDataIOSetting(
                name="u", members=[{"name": "u"}], is_time_series=True
            )
        ],
        label_settings=[
            ArrayDataIOSetting(
                name="w", members=[{"name": "w"}], is_time_series=True
            )
        ],
        directories=directories,
        window_setting=window_setting,
        allow_no_y_data=True,
    )

    assert len(dataset) == 5
    assert len(dataset[0].y_data) == 0


@pytest.mark.parametrize("dataset_cls", _DATASET_CLASSES)
def test__raise_error_when_number_of_windows_differ(
    dataset_cls: type[OnMemoryPhlowerDataSet | LazyPhlowerDataset],
    tmp_path: pathlib.Path,
):
    directories = _create_time_series(tmp_path, [5])
    window_setting = SlidingWindowForStage(
        inputs={"offset": 0, "size": 1, "stride": 1},
        labels={"offset": 0, "size": 2, "stride": 1},
        treat_as_samples=True,
    )

    with pytest.raises(ValueError, match="number of windows"):
        _create_dataset(dataset_cls, directories, window_setting)


@pytest.mark.parametrize("dataset_cls", _DATASET_CLASSES)
def test__raise_error_when_no_window(
    dataset_cls: type[OnMemoryPhlowerDataSet | LazyPhlowerDataset],
    tmp_path: pathlib.Path,
):
    directories = _create_time_series(tmp_path, [5, 2])
    window = {"offset": 0, "size": 3, "stride": 1}
    window_setting = SlidingWindowForStage(
        inputs=window, labels=window, treat_as_samples=True
    )

    with pytest.raises(ValueError, match="No window is found in data 1"):
        _create_dataset(dataset_cls, directories, window_setting)


@pytest.mark.parametrize(
    "dataset_cls, desired_msg",
    [
        (OnMemoryPhlowerDataSet, "Arrays for w are missing"),
        (LazyPhlowerDataset, "file not found for w"),
    ],
)
def test__raise_error_when_input_is_missing(
    dataset_cls: type[OnMemoryPhlowerDataSet | LazyPhlowerDataset],
    desired_msg: str,
    tmp_path: pathlib.Path,
):
    directories = _create_time_series(tmp_path, [5])
    window_setting = SlidingWindowForStage(treat_as_samples=True)

    with pytest.raises(ValueError, match=desired_msg):
        dataset_cls.create(
            input_settings=[
                ArrayDataIOSetting(
                    name="w", members=[{"name": "w"}], is_time_series=True
                )
            ],
            label_settings=[],
            directories=directories,
            window_setting=window_setting,
        )


@pytest.mark.parametrize("dataset_cls", _DATASET_CLASSES)
def test__raise_error_when_no_time_series(
    dataset_cls: type[OnMemoryPhlowerDataSet | LazyPhlowerDataset],
    tmp_path: pathlib.Path,
):
    directories = _create_time_series(tmp_path, [5])

    with pytest.raises(ValueError, match="needs time series"):
        dataset_cls.create(
            input_settings=[
                ArrayDataIOSetting(name="x", members=[{"name": "x"}])
            ],
            label_settings=[],
            directories=directories,
            window_setting=SlidingWindowForStage(treat_as_samples=True),
        )


@pytest.mark.parametrize(
    "n_data, n_windows_per_data, desired",
    [
        (3, None, [(0, None), (1, None), (2, None)]),
        (2, [2, 3], [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]),
        (3, [1, 1, 2], [(0, 0), (1, 0), (2, 0), (2, 1)]),
        (0, [], []),
    ],
)
def test__sample_indices(
    n_data: int,
    n_windows_per_data: list[int] | None,
    desired: list[tuple[int, int | None]],
):
    sample_indices = SampleIndices(
        n_data,
        None if n_windows_per_data is None else np.array(n_windows_per_data),
    )

    n_samples = len(desired)
    assert len(sample_indices) == n_samples
    assert [sample_indices[i] for i in range(n_samples)] == desired
    assert [sample_indices[i - n_samples] for i in range(n_samples)] == desired


@pytest.mark.parametrize(
    "n_data, n_windows_per_data, index",
    [(3, None, 3), (3, None, -4), (2, [2, 3], 5), (2, [2, 3], -6)],
)
def test__raise_error_when_sample_index_is_out_of_range(
    n_data: int, n_windows_per_data: list[int] | None, index: int
):
    sample_indices = SampleIndices(
        n_data,
        None if n_windows_per_data is None else np.array(n_windows_per_data),
    )

    with pytest.raises(IndexError, match=f"Index {index} is out of range"):
        sample_indices[index]


def test__raise_error_when_time_slicing_sparse_data():
    setting = ArrayDataIOSetting(
        name="a", members=[{"name": "a"}], is_time_series=True
    )

    with pytest.raises(ValueError, match="Time slicing of a needs a dense"):
        _slice_time(sp.csr_array(np.eye(3)), setting, slice(0, 1))
