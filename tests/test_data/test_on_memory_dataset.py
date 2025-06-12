import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
import yaml
from phlower._base import IPhlowerArray, phlower_array
from phlower.data import OnMemoryPhlowerDataSet
from phlower.settings import ModelIOSetting, PhlowerModelSetting
from phlower.utils.typing import ArrayDataType


@pytest.fixture
def create_loaded_data() -> list[dict[str, ArrayDataType]]:
    directory_names = ["data0", "data1", "data2"]
    name2dense_shape: dict[str, tuple[int, ...]] = {
        "x0": (1, 3, 4),
        "x1": (10, 5),
        "x2": (11, 3, 1),
        "x3": (11, 3),
        "y0": (1, 3, 4, 1),
        "y1": (1, 3, 4),
    }
    name2sparse_shape: dict[str, tuple[int, ...]] = {
        "s0": (5, 5),
        "s1": (10, 5),
    }

    results: list[dict[str, ArrayDataType]] = []
    for _ in directory_names:
        dict_data = {}
        for v_name, v_shape in name2dense_shape.items():
            dict_data[v_name] = phlower_array(np.random.rand(*v_shape))

        for v_name, v_shape in name2sparse_shape.items():
            dict_data[v_name] = phlower_array(sp.random(*v_shape, density=0.1))

        results.append(dict_data)
    return results


@pytest.mark.parametrize("yaml_file", ["sample1.yml", "sample2.yml"])
def test__onmemory_dataset_array_shape(
    yaml_file: str,
    create_loaded_data: list[dict[str, ArrayDataType]],
):
    yaml_path = pathlib.Path(__file__).parent / f"data/{yaml_file}"
    with open(yaml_path) as fr:
        content = yaml.safe_load(fr)

    setting = PhlowerModelSetting(**content["model"])
    dataset = OnMemoryPhlowerDataSet(
        loaded_data=create_loaded_data,
        input_settings=setting.inputs,
        label_settings=setting.labels,
        field_settings=setting.fields,
    )

    desired_shapes: dict[str, dict[str, list[int]]] = content["misc"]["tests"][
        "desired_shapes"
    ]

    for i in range(len(dataset)):
        item = dataset[i]

        for name, actual in item.x_data.items():
            assert actual.shape == tuple(desired_shapes["inputs"][name])

        for name, actual in item.y_data.items():
            assert actual.shape == tuple(desired_shapes["labels"][name])

        for name, actual in item.field_data.items():
            assert actual.shape == tuple(desired_shapes["fields"][name])


@pytest.mark.parametrize("yaml_file", ["sample_with_no_labels.yml"])
def test__on_memory_dataset_array_shape_when_no_ydata(
    yaml_file: str,
    create_loaded_data: list[dict[str, ArrayDataType]],
):
    yaml_path = pathlib.Path(__file__).parent / f"data/{yaml_file}"
    with open(yaml_path) as fr:
        content = yaml.safe_load(fr)

    setting = PhlowerModelSetting(**content["model"])
    dataset = OnMemoryPhlowerDataSet(
        loaded_data=create_loaded_data,
        input_settings=setting.inputs,
        label_settings=setting.labels,
        field_settings=setting.fields,
        allow_no_y_data=True,
    )

    desired_shapes: dict[str, dict[str, list[int]]] = content["misc"]["tests"][
        "desired_shapes"
    ]

    for i in range(len(dataset)):
        item = dataset[i]

        for name, actual in item.x_data.items():
            assert actual.shape == tuple(desired_shapes["inputs"][name])

        assert len(item.y_data) == 0

        for name, actual in item.field_data.items():
            assert actual.shape == tuple(desired_shapes["fields"][name])


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize(
    "data_type, expected",
    [
        ("input", ["x0", "x1", "x2", "x3"]),
        ("label", ["y0", "y1"]),
    ],
)
def test__get_members(
    index: int,
    data_type: str,
    expected: list[str],
    create_loaded_data: list[dict[str, ArrayDataType]],
):
    input_settings = [
        ModelIOSetting(
            name="all_x",
            members=[
                {"name": "x0"},
                {"name": "x1"},
                {"name": "x2"},
            ],
        ),
        ModelIOSetting(name="x3", members=[{"name": "x3"}]),
    ]
    label_settings = [
        ModelIOSetting(name="all_y", members=[{"name": "y0"}, {"name": "y1"}]),
        ModelIOSetting(name="y1", members=[{"name": "y1"}]),
    ]

    dataset = OnMemoryPhlowerDataSet(
        loaded_data=create_loaded_data,
        input_settings=input_settings,
        label_settings=label_settings,
    )

    members = dataset.get_members(index, data_type=data_type)
    assert len(members) == len(expected)

    for name in expected:
        assert name in members
        assert isinstance(members[name], IPhlowerArray)

        expected_arr = create_loaded_data[index][name]
        np.testing.assert_array_equal(
            members[name].to_numpy(), expected_arr.to_numpy()
        )
