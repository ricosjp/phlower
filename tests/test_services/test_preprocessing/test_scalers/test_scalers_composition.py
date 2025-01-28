import pathlib
import shutil

import numpy as np
import pytest
from phlower.io import PhlowerNumpyFile
from phlower.services.preprocessing._scalers import (
    PhlowerScalerWrapper,
    ScalersComposition,
)
from phlower.settings import PhlowerScalingSetting, PhlowerSetting


def test__from_setting():
    setting = PhlowerSetting.read_yaml(
        "tests/samples/random_dataset/preprocess.yml"
    )
    _ = ScalersComposition.from_setting(setting=setting)


@pytest.mark.parametrize(
    "name, allow_missing, method_name",
    [
        ("scaler_a", False, "identity"),
        ("scaler_b", False, "std_scale"),
        ("scaler_c", True, None),
    ],
)
def test__get_scaler(name: str, allow_missing: bool, method_name: str):
    name2scaler = {"scaler_a": "identity", "scaler_b": "std_scale"}
    scalers_dict = {k: PhlowerScalerWrapper(v) for k, v in name2scaler.items()}

    composition = ScalersComposition(scalers_dict)

    actual = composition.get_scaler(name, allow_missing=allow_missing)

    if method_name is None:
        assert actual is None

    else:
        assert actual.method_name == method_name


@pytest.fixture(scope="module")
def create_sample_dataset() -> list[PhlowerNumpyFile]:
    output_directory = pathlib.Path(__file__).parent / "tmp"

    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir()

    names = ["a", "b", "c"]
    for name in names:
        val = np.random.rand(3, 10)
        PhlowerNumpyFile.save(
            output_directory=output_directory,
            file_basename=f"val_{name}",
            data=val,
        )

    data_files = [
        PhlowerNumpyFile(output_directory / f"val_{name}.npy") for name in names
    ]
    return data_files


@pytest.mark.parametrize(
    "name2scaler",
    [
        (
            {
                "val_a": {"method": "min_max", "component_wise": True},
                "val_b": {"method": "std_scale"},
            }
        )
    ],
)
def test__retrieve_from_dumped_data(
    name2scaler: dict, create_sample_dataset: list[PhlowerNumpyFile]
):
    data_files = create_sample_dataset

    # fit to sample data
    setting = PhlowerSetting(
        scaling=PhlowerScalingSetting(variable_name_to_scalers=name2scaler)
    )
    composition = ScalersComposition.from_setting(setting)
    scaler_names = composition.get_scaler_names()

    for scaler_name in scaler_names:
        composition.lazy_partial_fit(
            scaler_name=scaler_name, data_files=data_files
        )

    dumped_data = composition.get_dumped_data()

    for name in scaler_names:
        scaler = composition.get_scaler(name)
        dumped_setting = dumped_data[name]

        assert scaler.method_name == dumped_setting.method
        assert scaler.componentwise == dumped_setting.component_wise
        assert dumped_setting.parameters["n_samples_seen_"] > 0
