import pathlib
import shutil
from collections import defaultdict

import numpy as np
import pytest
import scipy.sparse as sp
import yaml
from phlower.data import LazyPhlowerDataset
from phlower.io import PhlowerNumpyFile
from phlower.settings import PhlowerModelSetting
from phlower.utils.typing import ArrayDataType

_output_base_directory = pathlib.Path(__file__).parent / "tmp/datasets"


@pytest.fixture(scope="module")
def create_dataset() -> list[pathlib.Path]:
    if _output_base_directory.exists():
        shutil.rmtree(_output_base_directory)
    _output_base_directory.mkdir(parents=True)

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
        directories=[_output_base_directory / name for name in directories],
        field_settings=setting.fields,
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
        directories=[_output_base_directory / name for name in directories],
        field_settings=setting.fields,
        allow_no_y_data=True,
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
