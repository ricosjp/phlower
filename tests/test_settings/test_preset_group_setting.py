import pathlib

import pydantic
import pytest
import yaml

from phlower.io import PhlowerYamlFile
from phlower.settings import (
    GroupModuleSetting,
    PhlowerModelSetting,
    PresetGroupModuleSetting,
)
from phlower.utils.exceptions import (
    PhlowerModuleCycleError,
    PhlowerModuleDuplicateKeyError,
    PhlowerModuleKeyError,
    PhlowerModuleNodeDimSizeError,
)


def parse_file(file_name: str) -> dict:
    _TEST_DATA_DIR = pathlib.Path("tests/test_settings/data/presetgroups")
    with open(_TEST_DATA_DIR / file_name) as fr:
        data = yaml.load(fr, Loader=yaml.SafeLoader)

    return data


def _recursive_check(
    setting: GroupModuleSetting,
    desired: dict[str, int | dict[str, int | list[int]]],
):
    """check first n dims recursively"""

    for k, v in desired.items():
        if isinstance(v, dict):
            _recursive_check(setting.find_module(k), v)
            continue

        model = setting.find_module(k)
        assert isinstance(model, PresetGroupModuleSetting), f"{k} is not found."

        assert isinstance(v, list)
        resolved = [v.n_last_dim for v in model.inputs]
        assert resolved == v, f"n_last_dim mismatch at {k}."


@pytest.mark.parametrize(
    "file_name",
    [
        "simple_preset_group.yml",
        "can_omit_inputs.yml",
        "can_omit_inputs_dim.yml",
    ],
)
def test__can_resolve_phlower_networks(file_name: str, tmp_path: pathlib.Path):
    data = parse_file(file_name)

    setting = PhlowerModelSetting(**data["model"])
    setting.resolve()

    _recursive_check(setting.network, data["misc"]["tests"])

    # check serialization
    PhlowerYamlFile.save(
        output_directory=tmp_path,
        file_basename="setting",
        data=setting.model_dump(),
    )


@pytest.mark.parametrize("file_name", ["cycle_error.yml"])
def test__detect_cycle_error(file_name: str):
    data = parse_file(file_name)
    setting = PhlowerModelSetting(**data["model"])

    with pytest.raises(PhlowerModuleCycleError):
        setting.resolve()


@pytest.mark.parametrize("file_name", ["not_matched_last_node_error.yml"])
def test__detect_ndim_inconsistency(file_name: str):
    data = parse_file(file_name)
    setting = PhlowerModelSetting(**data["model"])

    with pytest.raises(PhlowerModuleNodeDimSizeError):
        setting.resolve()


@pytest.mark.parametrize("file_name", ["duplicate_keys_error.yml"])
def test__detect_duplicate_errors(file_name: str):
    data = parse_file(file_name)
    setting = PhlowerModelSetting(**data["model"])

    with pytest.raises(PhlowerModuleDuplicateKeyError):
        setting.resolve()


@pytest.mark.parametrize("file_name", ["key_missing_error.yml"])
def test__detect_key_missing_errors(file_name: str):
    data = parse_file(file_name)
    setting = PhlowerModelSetting(**data["model"])

    with pytest.raises(PhlowerModuleKeyError):
        setting.resolve()


@pytest.mark.parametrize(
    "file_name",
    [
        "cannot_omit_outputs.yml",
    ],
)
def test__raise_error_when_omitting_output_nodes(file_name: str):
    data = parse_file(file_name)
    with pytest.raises(pydantic.ValidationError, match="Field required"):
        PhlowerModelSetting(**data["model"])


@pytest.mark.parametrize("file_name", ["invalid_same_as_parameters.yml"])
def test__invalid_same_as_parameters(file_name: str):
    data = parse_file(file_name)

    with pytest.raises(
        NotImplementedError,
        match=(
            "PresetGroupModuleSetting does not support "
            "`nn_parameters_same_as` item."
        ),
    ):
        PhlowerModelSetting(**data["model"]).resolve()


@pytest.mark.parametrize("file_name", ["none_is_contained_in_members.yml"])
def test__raise_error_when_dimensions_contain_none(file_name: str):
    data = parse_file(file_name)
    setting = PhlowerModelSetting(**data["model"])

    with pytest.raises(
        ValueError, match="n_last_dim of feature0 cannot be determined"
    ):
        setting.resolve()


@pytest.mark.parametrize(
    "file_name",
    [
        "duplicate_inputs.yml",
        "duplicate_outputs.yml",
    ],
)
def test__duplicate_inputs_or_outputs_in_modules(file_name: str):
    data = parse_file(file_name)

    with pytest.raises(pydantic.ValidationError, match="duplicate keys exist"):
        PhlowerModelSetting(**data["model"])


@pytest.mark.parametrize(
    "file_name",
    [
        "extra_keywords.yml",
    ],
)
def test__raise_error_when_extra_keyword_exists(file_name: str):
    data = parse_file(file_name)

    with pytest.raises(
        pydantic.ValidationError, match="Extra inputs are not permitted"
    ):
        PhlowerModelSetting(**data["model"])


def test__raise_error_when_preset_type_not_exist():
    data = parse_file("not_exsit_preset_type.yml")

    with pytest.raises(
        ValueError, match="preset_type=NON_EXISTENT_PRESET is not implemented"
    ):
        PhlowerModelSetting(**data["model"])


def test__raise_error_when_preset_type_is_missing():
    data = parse_file("missing_preset_type.yml")

    with pytest.raises(
        ValueError, match="preset_type is not defined in Identity0"
    ):
        PhlowerModelSetting(**data["model"])


def test__raise_error_when_input_dim_is_invalid():
    data = parse_file("invalid_input_dim.yml")
    setting = PhlowerModelSetting(**data["model"])

    with pytest.raises(
        PhlowerModuleNodeDimSizeError,
        match="It is not consistent with the precedent modules",
    ):
        setting.resolve()
