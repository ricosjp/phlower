import pathlib

import numpy as np
import pytest
from phlower.data import LazyPhlowerDataset
from phlower.utils.typing import ArrayDataType


@pytest.mark.parametrize(
    "directories, desired",
    [(["data0", "data1"], 2), (["data0", "data1", "data2"], 3)],
)
def test__lazy_dataset_length(
    directories: list[str],
    desired: int,
    create_tmp_dataset: None,
    output_base_directory: pathlib.Path,
):
    directories = [output_base_directory / v for v in directories]
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
    x_variable_names: list[str],
    y_variable_names: list[str],
    support_names: list[str],
    directory_names: list[str],
    create_tmp_dataset: None,
    output_base_directory: pathlib.Path,
):
    directories = [output_base_directory / v for v in directory_names]
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
    x_variable_names: list[str],
    y_variable_names: list[str],
    support_names: list[str],
    directory_names: list[str],
    create_tmp_dataset: None,
    output_base_directory: pathlib.Path,
):
    directories = [output_base_directory / v for v in directory_names]
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
