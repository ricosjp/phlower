import pathlib

import pytest
import yaml

from phlower.settings import GroupModuleSetting
from phlower.utils.exceptions import (
    PhlowerModuleCycleError,
    PhlowerModuleDuplicateKeyError,
    PhlowerModuleKeyError,
    PhlowerModuleNodeDimSizeError,
)


def parse_file(file_name: str) -> dict:
    _TEST_DATA_DIR = pathlib.Path("tests/test_settings/data")
    with open(_TEST_DATA_DIR / file_name) as fr:
        data = yaml.load(fr, Loader=yaml.SafeLoader)

    return data


@pytest.mark.parametrize(
    "file_name", ["simple_module.yml", "simple_group_in_group.yml"]
)
def test__can_resolve_phlower_networks(file_name):
    data = parse_file(file_name)

    setting = GroupModuleSetting(**data["model"])
    setting.resolve(is_first=True)


@pytest.mark.parametrize("file_name", ["cycle_error.yml"])
def test__detect_cycle_error(file_name):
    data = parse_file(file_name)
    setting = GroupModuleSetting(**data["model"])

    with pytest.raises(PhlowerModuleCycleError):
        setting.resolve(is_first=True)


@pytest.mark.parametrize("file_name", ["not_matched_last_node_error.yml"])
def test__detect_ndim_inconsistency(file_name):
    data = parse_file(file_name)
    setting = GroupModuleSetting(**data["model"])

    with pytest.raises(PhlowerModuleNodeDimSizeError):
        setting.resolve(is_first=True)


@pytest.mark.parametrize("file_name", ["duplicate_keys_error.yml"])
def test__detect_duplicate_errors(file_name):
    data = parse_file(file_name)
    setting = GroupModuleSetting(**data["model"])

    with pytest.raises(PhlowerModuleDuplicateKeyError):
        setting.resolve(is_first=True)


@pytest.mark.parametrize("file_name", ["key_missing_error.yml"])
def test__detect_key_missing(file_name):
    data = parse_file(file_name)
    setting = GroupModuleSetting(**data["model"])

    with pytest.raises(PhlowerModuleKeyError):
        setting.resolve(is_first=True)
