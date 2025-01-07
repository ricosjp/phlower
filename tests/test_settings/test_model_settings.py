import pathlib

import pytest
import yaml
from phlower.settings import (
    GroupModuleSetting,
    ModuleSetting,
    PhlowerModelSetting,
)
from phlower.utils.exceptions import (
    PhlowerIterationSolverSettingError,
    PhlowerModuleCycleError,
    PhlowerModuleDuplicateKeyError,
    PhlowerModuleKeyError,
    PhlowerModuleNodeDimSizeError,
)


def parse_file(file_name: str) -> dict:
    _TEST_DATA_DIR = pathlib.Path("tests/test_settings/data/models")
    with open(_TEST_DATA_DIR / file_name) as fr:
        data = yaml.load(fr, Loader=yaml.SafeLoader)

    return data


def _recursive_check(
    setting: GroupModuleSetting, desired: dict[str, int | dict[str, int]]
):
    """check first n dims recursively"""

    for k, v in desired.items():
        if isinstance(v, dict):
            _recursive_check(setting.find_module(k), v)
            continue

        model = setting.find_module(k)
        assert isinstance(model, ModuleSetting)

        assert model.nn_parameters.get_n_nodes()[0] == v


@pytest.mark.parametrize(
    "file_name", ["simple_module.yml", "simple_group_in_group.yml"]
)
def test__can_resolve_phlower_networks(file_name: str):
    data = parse_file(file_name)

    setting = PhlowerModelSetting(**data["model"])
    setting.resolve()

    _recursive_check(setting.network, data["misc"]["tests"])


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
def test__detect_key_missing(file_name: str):
    data = parse_file(file_name)
    setting = PhlowerModelSetting(**data["model"])

    with pytest.raises(PhlowerModuleKeyError):
        setting.resolve()


@pytest.mark.parametrize(
    "file_name",
    [
        "target_missing_solver_1.yml",
        "target_missing_solver_2.yml",
        "target_missing_solver_3.yml",
    ],
)
def test__check_solver_target_name_is_missing(file_name: str):
    data = parse_file(file_name)
    setting = PhlowerModelSetting(**data["model"])

    with pytest.raises(PhlowerIterationSolverSettingError):
        setting.resolve()


def _recursive_check_inputs_and_outputs(
    setting: GroupModuleSetting | ModuleSetting,
    desired: dict[str, dict[str, list]],
):
    name = setting.name
    answer = desired[name]

    if isinstance(setting, GroupModuleSetting):
        desired_inputs = answer["inputs"]
        desired_outputs = answer["outputs"]

        for actual in setting.inputs:
            assert actual.name in desired_inputs
            assert actual.n_last_dim == desired_inputs[actual.name]

        for actual in setting.outputs:
            assert actual.name in desired_outputs
            assert actual.n_last_dim == desired_outputs[actual.name]

        for module in setting.modules:
            _recursive_check_inputs_and_outputs(module, desired)

    else:
        desired_inputs = answer.get("input_keys")
        desired_outputs = answer.get("output_key")

        assert setting.input_keys == desired_inputs
        assert setting.output_key == desired_outputs


@pytest.mark.parametrize(
    "file_name",
    [
        "set_naming_1.yml",
        "set_naming_2.yml",
    ],
)
def test__set_automatically_inputs_and_outpus(file_name: str):
    data = parse_file(file_name)
    setting = PhlowerModelSetting(**data["model"])
    setting.resolve()

    desired = data["misc"]["tests"]

    _recursive_check_inputs_and_outputs(setting.network, desired)


@pytest.mark.parametrize(
    "file_name",
    [
        "cannot_omit_inputs.yml",
    ],
)
def test__raise_error_when_omitting_inputs_in_the_head_modules(file_name: str):
    data = parse_file(file_name)
    setting = PhlowerModelSetting(**data["model"])

    with pytest.raises(ValueError) as ex:
        setting.resolve()

    assert "only one input is allowed" in str(ex.value)


@pytest.mark.parametrize("file_name", ["none_is_contained_in_members.yml"])
def test__raise_error_when_dimensions_contain_none(file_name: str):
    data = parse_file(file_name)
    setting = PhlowerModelSetting(**data["model"])

    with pytest.raises(ValueError) as ex:
        setting.resolve()

    assert "n_last_dim of feature0 cannot be determined" in str(ex.value)


@pytest.mark.parametrize("file_name", ["simple_module_time_slice.yml"])
def test__input_time_series(file_name: str):
    data = parse_file(file_name)
    setting = PhlowerModelSetting(**data["model"])

    desired_inputs_slice = data["misc"]["tests"]["inputs"]
    for item in setting.inputs:
        if desired_inputs_slice[item.name] is None:
            assert item.time_slice is None
            continue

        assert item.time_slice_object == slice(*desired_inputs_slice[item.name])

    desired_labels_slice = data["misc"]["tests"]["labels"]
    for item in setting.labels:
        if desired_labels_slice[item.name] is None:
            assert item.time_slice is None
            continue

        assert item.time_slice_object == slice(*desired_labels_slice[item.name])
