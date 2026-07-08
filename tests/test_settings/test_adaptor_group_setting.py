import pathlib

import pytest

from phlower.settings import PhlowerSetting

_DATA_DIR = pathlib.Path(__file__).parent / "data/adaptor_groups"


@pytest.mark.parametrize(
    "file_name, expected_msg",
    [
        ("invalid_keys.yml", "is not passed to MAIN_GROUP"),
        ("not_match_prefix.yml", "does not match any input key"),
        ("multiple_match_prefix.yml", "Multiple prefixes matched for key"),
        ("not_match_input_keys.yml", "is not matched with any prefix"),
        ("duplicate_prefixes.yml", "Duplicate prefix is detected"),
        ("empty_prefix.yml", "Please set at least one prefix"),
    ],
)
def test__detect_invalid_keys(file_name: str, expected_msg: str):
    file_path = _DATA_DIR / file_name

    with pytest.raises(ValueError, match=expected_msg):
        setting = PhlowerSetting.read_yaml(file_path)
        setting.model.resolve()


@pytest.mark.parametrize("file_name", ["pinn_sample.yml"])
def test__load_yaml_file(file_name: str):
    file_path = _DATA_DIR / file_name
    setting = PhlowerSetting.read_yaml(file_path)
    setting.model.resolve()
