import logging
import pathlib
import shutil

import pytest
from phlower.io import PhlowerYamlFile
from phlower.settings import (
    PhlowerModelSetting,
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


@pytest.mark.parametrize(
    "yaml_file", ["tests/test_settings/data/e2e/too_high_version.yml"]
)
def test__raise_warnings_when_version_is_not_compatible(
    yaml_file: str,
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level(logging.WARNING, logger="phlower"):
        _ = PhlowerSetting.read_yaml(yaml_file)

    assert len(caplog.records) > 0
    for record in caplog.records:
        assert record.levelname == "WARNING"
    assert "Version number of input setting file is higher" in caplog.text
