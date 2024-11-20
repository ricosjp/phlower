import pathlib

import numpy as np
import pytest
from phlower.data import LazyPhlowerDataset
from phlower.settings import ModelIOSetting
from phlower.utils.typing import ArrayDataType


def _to_modelIO_settings(
    names: list[tuple[str, int]] | None,
) -> list[ModelIOSetting] | None:
    if names is None:
        return None
    return [
        ModelIOSetting(name=v, members=[{"name": v, "n_last_dim": n_dim}])
        for v, n_dim in names
    ]


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
        input_settings=_to_modelIO_settings([("x0", 1), ("x1", 1)]),
        label_settings=_to_modelIO_settings([("y0", 1)]),
        directories=directories,
        field_settings=_to_modelIO_settings([("s0", 1)]),
    )
    assert len(dataset) == desired


@pytest.mark.parametrize(
    "x_variables, y_variables, field_names, directory_names",
    [
        (
            [("x0", 1), ("x1", 1), ("x2", 1)],
            [("y0", 1)],
            [("s0", None), ("s1", None)],
            ["data0", "data1", "data2"],
        )
    ],
)
def test__lazy_dataset_getitem(
    x_variables: list[str],
    y_variables: list[str],
    field_names: list[str],
    directory_names: list[str],
    create_tmp_dataset: None,
    output_base_directory: pathlib.Path,
):
    directories = [output_base_directory / v for v in directory_names]
    dataset = LazyPhlowerDataset(
        input_settings=_to_modelIO_settings(x_variables),
        label_settings=_to_modelIO_settings(y_variables),
        directories=directories,
        field_settings=_to_modelIO_settings(field_names),
    )

    assert len(dataset) > 1
    desired: dict[str, dict[str, ArrayDataType]] = create_tmp_dataset
    for i in range(len(dataset)):
        item = dataset[i]
        data_name = item.data_directory.path.name

        assert data_name in desired

        for v_name, _ in x_variables:
            desired_shape = desired[data_name][v_name].shape
            np.testing.assert_array_almost_equal(
                item.x_data[v_name].to_numpy().reshape(desired_shape),
                desired[data_name][v_name],
            )

        for v_name, _ in y_variables:
            desired_shape = desired[data_name][v_name].shape
            np.testing.assert_array_almost_equal(
                item.y_data[v_name].to_numpy().reshape(desired_shape),
                desired[data_name][v_name],
            )

        for v_name, _ in field_names:
            np.testing.assert_array_almost_equal(
                item.field_data[v_name].to_numpy().todense(),
                desired[data_name][v_name].todense(),
            )


@pytest.mark.parametrize(
    "x_variables, y_variables, field_names, directory_names",
    [
        (
            [("x0", 1), ("x1", 1), ("x2", 1)],
            None,
            [("s0", None), ("s1", None)],
            ["data0", "data1", "data2"],
        ),
        (
            [("x0", 1), ("x1", 1), ("x2", 1)],
            [("y3", 1)],
            [("s0", None), ("s1", None)],
            ["data0", "data1", "data2"],
        ),
    ],
)
def test__lazy_dataset_getitem_when_no_ydata(
    x_variables: list[str],
    y_variables: list[str],
    field_names: list[str],
    directory_names: list[str],
    create_tmp_dataset: None,
    output_base_directory: pathlib.Path,
):
    directories = [output_base_directory / v for v in directory_names]
    dataset = LazyPhlowerDataset(
        input_settings=_to_modelIO_settings(x_variables),
        label_settings=_to_modelIO_settings(y_variables),
        directories=directories,
        field_settings=_to_modelIO_settings(field_names),
        allow_no_y_data=True,
    )
    assert len(dataset) > 1

    for i in range(len(dataset)):
        item = dataset[i]
        assert len(item.y_data) == 0
