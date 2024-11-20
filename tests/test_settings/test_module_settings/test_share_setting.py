import pathlib
from unittest import mock

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import given, settings
from phlower.settings import PhlowerModelSetting, PhlowerSetting
from phlower.settings._module_settings import ShareSetting


@given(st.lists(st.integers(), max_size=100))
@settings(max_examples=100)
def test__gather_input_dims_of_reference_setting(input_dims: list[int]):
    setting = ShareSetting(reference_name="dummy")
    mocked = mock.MagicMock()
    setting.reference = mocked

    _ = setting.gather_input_dims(*input_dims)

    mocked.gather_input_dims.assert_called_once_with(*input_dims)


def test__get_n_nodes_of_reference_setting():
    setting = ShareSetting(reference_name="dummy")
    mocked = mock.MagicMock()
    setting.reference = mocked

    _ = setting.get_n_nodes()

    mocked.get_n_nodes.assert_called_once()


@pytest.mark.parametrize("reference_name", ["dummy", "mlp0", "gcn0"])
def test__call_parent_function_when_get_reference(reference_name: str):
    setting = ShareSetting(reference_name=reference_name)
    mocked = mock.MagicMock()

    setting.get_reference(mocked)

    mocked.search_module_setting.assert_called_once_with(reference_name)


# region E2E tests only for ShareSettings

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/share_setting"


def test__can_resolve():
    setting = PhlowerSetting.read_yaml(_TEST_DATA_DIR / "with_share_nn.yml")
    setting.model.network.resolve(is_first=True)


@pytest.mark.parametrize(
    "yaml_file", ["check_gcn_share_nodes.yml", "check_mlp_share_nodes.yml"]
)
def test__nodes_after_resolve(yaml_file: str):
    with open(_TEST_DATA_DIR / yaml_file) as fr:
        content = yaml.load(fr, Loader=yaml.SafeLoader)

    setting = PhlowerModelSetting(**content["model"])
    setting.network.resolve(is_first=True)

    assert len(content["misc"]["tests"].items()) > 0

    for key, value in content["misc"]["tests"].items():
        target = setting.network.search_module_setting(key)
        assert target.get_n_nodes() == value


# endregion
