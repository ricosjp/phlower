import pathlib
import shutil
from collections import defaultdict

import numpy as np
import pytest
import scipy.sparse as sp

from phlower.data import LazyPhlowerDataset
from phlower.io import PhlowerNumpyFile
from phlower.utils.typing import ArrayDataType

_output_base_directory = pathlib.Path(__file__).parent / "tmp/datasets"


@pytest.fixture(scope="module")
def create_tmp_dataset():
    if _output_base_directory.exists():
        shutil.rmtree(_output_base_directory)
    _output_base_directory.mkdir(parents=True)

    directory_names = ["data0", "data1", "data2"]
    name2dense_shape: dict[str, tuple[int, ...]] = {
        "x0": (1, 3, 4),
        "x1": (10, 5),
        "x2": (11, 3),
        "y0": (1, 3, 4),
    }
    name2sparse_shape: dict[str, tuple[int, ...]] = {
        "s0": (5, 5),
        "s1": (10, 5),
    }

    results: dict[str, dict[str, ArrayDataType]] = defaultdict(dict)
    for name in directory_names:
        _output_directory = _output_base_directory / name
        _output_directory.mkdir()

        for v_name, v_shape in name2dense_shape.items():
            arr = np.random.rand(*v_shape)
            PhlowerNumpyFile.save(_output_directory, v_name, arr)
            results[name][v_name] = arr

        for v_name, v_shape in name2sparse_shape.items():
            arr = sp.random(*v_shape, density=0.1)
            PhlowerNumpyFile.save(_output_directory, v_name, arr)
            results[name][v_name] = arr

    return results


@pytest.mark.parametrize(
    "directories, desired",
    [(["data0", "data1"], 2), (["data0", "data1", "data2"], 3)],
)
def test__lazy_dataset_length(directories, desired, create_tmp_dataset):
    directories = [_output_base_directory / v for v in directories]
    dataset = LazyPhlowerDataset(
        x_variable_names=["x0", "x1"],
        y_variable_names=["y0"],
        directories=directories,
        support_names=["s0"],
    )
    assert len(dataset) == desired


@pytest.mark.parametrize(
    "x_variable_names, y_variable_names, support_names, directory_names",
    [(["x0", "x1", "x2"], ["y0"], ["s0", "s1"], ["data0", "data1", "data2"])],
)
def test__lazy_dataset_getitem(
    x_variable_names,
    y_variable_names,
    support_names,
    directory_names,
    create_tmp_dataset,
):
    directories = [_output_base_directory / v for v in directory_names]
    dataset = LazyPhlowerDataset(
        x_variable_names=x_variable_names,
        y_variable_names=y_variable_names,
        directories=directories,
        support_names=support_names,
    )

    assert len(dataset) > 1
    desired: dict[str, dict[str, ArrayDataType]] = create_tmp_dataset
    for i in range(len(dataset)):
        item = dataset[i]
        data_name = item.data_directory.path.name

        assert data_name in desired

        for v_name in x_variable_names:
            np.testing.assert_array_almost_equal(
                item.x_data[v_name].to_numpy(), desired[data_name][v_name]
            )

        for v_name in y_variable_names:
            np.testing.assert_array_almost_equal(
                item.y_data[v_name].to_numpy(), desired[data_name][v_name]
            )

        for v_name in support_names:
            np.testing.assert_array_almost_equal(
                item.sparse_supports[v_name].to_numpy().todense(),
                desired[data_name][v_name].todense(),
            )


@pytest.mark.parametrize(
    "x_variable_names, y_variable_names, support_names, directory_names",
    [
        (["x0", "x1", "x2"], None, ["s0", "s1"], ["data0", "data1", "data2"]),
        (["x0", "x1", "x2"], ["y3"], ["s0", "s1"], ["data0", "data1", "data2"]),
    ],
)
def test__lazy_dataset_getitem_when_no_ydata(
    x_variable_names,
    y_variable_names,
    support_names,
    directory_names,
    create_tmp_dataset,
):
    directories = [_output_base_directory / v for v in directory_names]
    dataset = LazyPhlowerDataset(
        x_variable_names=x_variable_names,
        y_variable_names=y_variable_names,
        directories=directories,
        support_names=support_names,
        allow_no_y_data=True,
    )
    assert len(dataset) > 1

    for i in range(len(dataset)):
        item = dataset[i]
        assert len(item.y_data) == 0
