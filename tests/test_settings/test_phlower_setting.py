import pathlib
import shutil

import pydantic
import pytest
from phlower.io import PhlowerYamlFile
from phlower.settings import (
    PhlowerModelSetting,
    PhlowerPredictorSetting,
    PhlowerSetting,
)


@pytest.mark.parametrize(
    "variable_dimensions",
    [({"vala": {"mass_": 1, "time": 4}}), ({"sample": {"kg": 2, "Am": -2}})],
)
def test__invalid_dimension_model_setting(variable_dimensions: dict):
    with pytest.raises(TypeError):
        _ = PhlowerModelSetting(
            variable_dimensions=variable_dimensions,
            network={
                "name": "sample",
                "inputs": [],
                "outputs": [],
                "modules": [],
            },
        )


@pytest.mark.parametrize(
    "selection_mode", ["best", "latest", "train_best", "specified"]
)
def test__valid_selection_mode(selection_mode: str):
    _ = PhlowerPredictorSetting(selection_mode=selection_mode)


@pytest.mark.parametrize("selection_mode", ["other", "best_of_best"])
def test__invalid_selection_mode(selection_mode: str):
    with pytest.raises(pydantic.ValidationError):
        _ = PhlowerPredictorSetting(selection_mode=selection_mode)


def test__model_dump():
    setting = PhlowerSetting.read_yaml(
        "tests/test_settings/data/e2e/setting1.yml"
    )

    output_directory = pathlib.Path("tests/test_settings/tmp")
    shutil.rmtree(output_directory, ignore_errors=True)
    _ = PhlowerYamlFile.save(
        output_directory=output_directory,
        file_basename="output",
        data=setting.model_dump(),
    )

    setting2 = PhlowerSetting.read_yaml("tests/test_settings/tmp/output.yml")

    assert setting.model_dump_json() == setting2.model_dump_json()
