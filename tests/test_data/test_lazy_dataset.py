import pathlib
import secrets
import shutil
from collections import defaultdict

import numpy as np
import pytest
import scipy.sparse as sp
import yaml
from phlower_tensor import IPhlowerArray

from phlower.data import LazyPhlowerDataset
from phlower.io import PhlowerNumpyFile
from phlower.settings import ModelIOSetting, PhlowerModelSetting
from phlower.utils.typing import ArrayDataType

TEST_ENCRYPT_KEY = secrets.token_bytes(32)
OUTPUT_BASE_DIRECTORY = pathlib.Path(__file__).parent / "tmp/datasets"


@pytest.fixture(scope="module")
def create_dataset() -> list[pathlib.Path]:
    if OUTPUT_BASE_DIRECTORY.exists():
        shutil.rmtree(OUTPUT_BASE_DIRECTORY)
    OUTPUT_BASE_DIRECTORY.mkdir(parents=True)

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

    results: dict[str, dict[str, ArrayDataType]] = defaultdict(dict)
    for name in directory_names:
        _output_directory = OUTPUT_BASE_DIRECTORY / name
        _output_directory.mkdir()

        for v_name, v_shape in name2dense_shape.items():
            arr = np.random.rand(*v_shape)
            PhlowerNumpyFile.save(
                _output_directory, v_name, arr, encrypt_key=TEST_ENCRYPT_KEY
            )
            results[name][v_name] = arr

        for v_name, v_shape in name2sparse_shape.items():
            arr = sp.random(*v_shape, density=0.1)
            PhlowerNumpyFile.save(
                _output_directory, v_name, arr, encrypt_key=TEST_ENCRYPT_KEY
            )
            results[name][v_name] = arr


@pytest.mark.parametrize("yaml_file", ["sample1.yml", "sample2.yml"])
@pytest.mark.parametrize(
    "directories",
    [(["data0", "data1"]), (["data0", "data1", "data2"]), (["data0"])],
)
def test__lazy_dataset_array_shape(
    yaml_file: str, directories: list[str], create_dataset: None
):
    yaml_path = pathlib.Path(__file__).parent / f"data/{yaml_file}"
    with open(yaml_path) as fr:
        content = yaml.safe_load(fr)

    setting = PhlowerModelSetting(**content["model"])
    dataset = LazyPhlowerDataset(
        input_settings=setting.inputs,
        label_settings=setting.labels,
        directories=[OUTPUT_BASE_DIRECTORY / name for name in directories],
        field_settings=setting.fields,
        decrypt_key=TEST_ENCRYPT_KEY,
    )

    desired_shapes: dict[str, dict[str, list[int]]] = content["misc"]["tests"][
        "desired_shapes"
    ]

    assert len(dataset) == len(directories)
    for i in range(len(dataset)):
        item = dataset[i]

        for name, actual in item.x_data.items():
            assert actual.shape == tuple(desired_shapes["inputs"][name])

        for name, actual in item.y_data.items():
            assert actual.shape == tuple(desired_shapes["labels"][name])

        for name, actual in item.field_data.items():
            assert actual.shape == tuple(desired_shapes["fields"][name])


@pytest.mark.parametrize("yaml_file", ["sample_with_no_labels.yml"])
@pytest.mark.parametrize(
    "directories",
    [(["data0", "data1"]), (["data0", "data1", "data2"]), (["data0"])],
)
def test__lazy_dataset_array_shape_when_no_ydata(
    yaml_file: str, directories: list[str], create_dataset: None
):
    yaml_path = pathlib.Path(__file__).parent / f"data/{yaml_file}"
    with open(yaml_path) as fr:
        content = yaml.safe_load(fr)

    setting = PhlowerModelSetting(**content["model"])
    dataset = LazyPhlowerDataset(
        input_settings=setting.inputs,
        label_settings=setting.labels,
        directories=[OUTPUT_BASE_DIRECTORY / name for name in directories],
        field_settings=setting.fields,
        allow_no_y_data=True,
        decrypt_key=TEST_ENCRYPT_KEY,
    )

    desired_shapes: dict[str, dict[str, list[int]]] = content["misc"]["tests"][
        "desired_shapes"
    ]

    assert len(dataset) == len(directories)
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
    index: int, data_type: str, expected: list[str], create_dataset: None
):
    directories = [
        OUTPUT_BASE_DIRECTORY / "data0",
        OUTPUT_BASE_DIRECTORY / "data1",
        OUTPUT_BASE_DIRECTORY / "data2",
    ]
    input_settings = [
        {
            "name": "all_x0",
            "members": [
                {"name": "x0", "n_last_dim": 1},
                {"name": "x1", "n_last_dim": 1},
                {"name": "x2", "n_last_dim": 1},
            ],
        },
        {
            "name": "x3",
            "members": [
                {"name": "x3", "n_last_dim": 1},
            ],
        },
        {"name": "x1", "members": [{"name": "x1", "n_last_dim": 1}]},
    ]
    label_settings = [
        {
            "name": "all_y",
            "members": [
                {"name": "y0", "n_last_dim": 1},
                {"name": "y1", "n_last_dim": 1},
            ],
        },
    ]
    field_settings = [
        {"name": "s0", "members": [{"name": "s0", "n_last_dim": None}]}
    ]

    dataset = LazyPhlowerDataset(
        input_settings=[ModelIOSetting(**v) for v in input_settings],
        label_settings=[ModelIOSetting(**v) for v in label_settings],
        directories=directories,
        field_settings=[ModelIOSetting(**v) for v in field_settings],
        decrypt_key=TEST_ENCRYPT_KEY,
    )

    members = dataset.get_members(index, data_type=data_type)
    assert len(members) == len(expected)

    for name in expected:
        assert name in members
        assert isinstance(members[name], IPhlowerArray)

        expected_arr = PhlowerNumpyFile(
            directories[index] / f"{name}.npy.enc"
        ).load(decrypt_key=TEST_ENCRYPT_KEY)
        np.testing.assert_array_equal(
            members[name].to_numpy(), expected_arr.to_numpy()
        )
