import pathlib
import sys

import numpy as np
import pytest
import yaml
from phlower_tensor import IPhlowerArray

from phlower.data import OnMemoryPhlowerDataSet
from phlower.io import PhlowerNumpyFile
from phlower.settings import ArrayDataIOSetting, PhlowerModelSetting
from phlower.utils._extended_simulation_field import PyVistaMeshAdapter


@pytest.mark.parametrize("yaml_file", ["sample1.yml", "sample2.yml"])
def test__onmemory_dataset_array_shape(
    yaml_file: str,
    create_encrypted_dataset: bytes,
    output_base_directory: pathlib.Path,
):
    encrypt_key = create_encrypted_dataset
    yaml_path = pathlib.Path(__file__).parent / f"data/{yaml_file}"
    with open(yaml_path) as fr:
        content = yaml.safe_load(fr)

    directories = list(output_base_directory.iterdir())

    setting = PhlowerModelSetting(**content["model"])
    dataset = OnMemoryPhlowerDataSet.create(
        input_settings=setting.inputs,
        label_settings=setting.labels,
        field_settings=setting.fields,
        directories=directories,
        decrypt_key=encrypt_key,
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
    create_encrypted_dataset: bytes,
    output_base_directory: pathlib.Path,
):
    encrypt_key = create_encrypted_dataset
    yaml_path = pathlib.Path(__file__).parent / f"data/{yaml_file}"
    with open(yaml_path) as fr:
        content = yaml.safe_load(fr)

    directories = list(output_base_directory.iterdir())
    setting = PhlowerModelSetting(**content["model"])
    dataset = OnMemoryPhlowerDataSet.create(
        input_settings=setting.inputs,
        label_settings=setting.labels,
        field_settings=setting.fields,
        allow_no_y_data=True,
        directories=directories,
        decrypt_key=encrypt_key,
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
    create_encrypted_dataset: bytes,
    output_base_directory: pathlib.Path,
):
    encrypt_key = create_encrypted_dataset
    input_settings = [
        ArrayDataIOSetting(
            name="all_x",
            members=[
                {"name": "x0"},
                {"name": "x1"},
                {"name": "x2"},
            ],
        ),
        ArrayDataIOSetting(name="x3", members=[{"name": "x3"}]),
    ]
    label_settings = [
        ArrayDataIOSetting(
            name="all_y", members=[{"name": "y0"}, {"name": "y1"}]
        ),
        ArrayDataIOSetting(name="y1", members=[{"name": "y1"}]),
    ]

    directories = list(output_base_directory.iterdir())
    dataset = OnMemoryPhlowerDataSet.create(
        input_settings=input_settings,
        label_settings=label_settings,
        decrypt_key=encrypt_key,
        directories=directories,
    )

    members = dataset.get_members(index, data_type=data_type)
    assert len(members) == len(expected)

    for name in expected:
        assert name in members
        assert isinstance(members[name], IPhlowerArray)

        expected_arr = PhlowerNumpyFile(
            directories[index] / f"{name}.npy.enc"
        ).load(decrypt_key=encrypt_key)
        np.testing.assert_array_equal(
            members[name].to_numpy(), expected_arr.to_numpy()
        )


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Requires Python 3.12 or higher for graphlow.",
)
@pytest.mark.parametrize("yaml_file", ["sample1_with_mesh.yml"])
def test__onmemory_dataset_array_with_mesh(
    yaml_file: str,
    create_encrypted_dataset: bytes,
    output_base_directory: pathlib.Path,
):
    encrypt_key = create_encrypted_dataset
    yaml_path = pathlib.Path(__file__).parent / f"data/{yaml_file}"
    with open(yaml_path) as fr:
        content = yaml.safe_load(fr)

    directories = list(output_base_directory.iterdir())

    setting = PhlowerModelSetting(**content["model"])
    dataset = OnMemoryPhlowerDataSet.create(
        input_settings=setting.inputs,
        label_settings=setting.labels,
        field_settings=setting.fields,
        directories=directories,
        decrypt_key=encrypt_key,
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

        for _, actual in item.field_data.items():
            assert isinstance(actual, PyVistaMeshAdapter)
            for val_name in desired_shapes["fields"].keys():
                assert val_name in actual.get_pvmesh().point_data
