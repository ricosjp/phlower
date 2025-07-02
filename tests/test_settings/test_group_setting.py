"""
The tests in this file is target only for validation
 when creating GroupModuleSetting instance.
In other others, if you want to write tests
 for validation whe calling `resolve`,
 please write them in `test_model_settings.py`
"""

import pathlib

import pydantic
import pytest
import yaml
from phlower.settings import GroupModuleSetting


def parse_file(file_name: str) -> dict:
    _TEST_DATA_DIR = pathlib.Path("tests/test_settings/data/groups")
    with open(_TEST_DATA_DIR / file_name) as fr:
        data = yaml.load(fr, Loader=yaml.SafeLoader)

    return data


@pytest.mark.parametrize("solver_type", ["newton", "aaa", "bbb"])
def test__not_implemeted_iteration_solver_name(solver_type: str):
    with pytest.raises(NotImplementedError) as ex:
        _ = GroupModuleSetting(
            name="Sample",
            inputs=[],
            outputs=[],
            modules=[],
            destinations=[],
            solver_type=solver_type,
        )
    assert f"solver_type: {solver_type} is not implemented." in str(ex)


@pytest.mark.parametrize(
    "file_name",
    [
        "extra_keywords.yml",
    ],
)
def test__raise_error_when_extra_keyword_exists(file_name: str):
    data = parse_file(file_name)

    with pytest.raises(pydantic.ValidationError) as ex:
        _ = GroupModuleSetting(**data["network"])

    assert "Extra inputs are not permitted" in str(ex.value)


@pytest.mark.parametrize(
    "file_name",
    [
        "duplicate_inputs.yml",
        "duplicate_outputs.yml",
    ],
)
def test__duplicate_inputs_or_outputs_in_modules(file_name: str):
    data = parse_file(file_name)

    with pytest.raises(pydantic.ValidationError) as ex:
        _ = GroupModuleSetting(**data["network"])

    assert "duplicate keys exist" in str(ex.value)


@pytest.mark.parametrize(
    "file_name",
    [
        "duplicate_modules.yml",
    ],
)
def test__duplicate_modules_error(file_name: str):
    data = parse_file(file_name)

    with pytest.raises(pydantic.ValidationError) as ex:
        _ = GroupModuleSetting(**data["network"])

    assert "Duplicate module name" in str(ex.value)


@pytest.mark.parametrize(
    "time_series_length, success",
    [
        (-4, False),
        (-100, False),
        (-1, True),  # -1 is allowed for time series length
        (0, True),
        (5, True),
        (10, True),
    ],
)
def test__raise_error_when_invalid_time_series_length(
    time_series_length: int, success: bool
):
    data = parse_file("time_series_length.yml")
    data["network"]["time_series_length"] = time_series_length

    if success:
        setting = GroupModuleSetting(**data["network"])
        assert setting.time_series_length == time_series_length
    else:
        with pytest.raises(pydantic.ValidationError) as ex:
            _ = GroupModuleSetting(**data["network"])

        assert "time_series_length must be -1 or larger" in str(ex.value)
