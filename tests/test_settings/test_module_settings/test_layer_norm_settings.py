import pathlib

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import given, settings

from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import LayerNormSetting


@pytest.mark.parametrize(
    "nodes",
    [([16, 16]), ([-1, 50]), ([10, 10]), ([7, 7])],
)
def test__can_accept_valid_n_nodes(nodes: list[int] | None):
    _ = LayerNormSetting(nodes=nodes)


@pytest.mark.parametrize(
    "nodes",
    [([5]), ([10, 10, 10]), ([16, 8]), ([4, 8]), ([30, -20]), ([-1, -1])],
)
def test__raise_error_when_invalid_n_nodes(nodes: list[int]):
    with pytest.raises(ValueError):
        _ = LayerNormSetting(nodes=nodes)


@pytest.mark.parametrize("input_dims", [([30]), ([40]), ([100])])
def test__gather_input_dims(input_dims: list[int]):
    setting = LayerNormSetting()

    assert setting.gather_input_dims(*input_dims) == input_dims[0]


@pytest.mark.parametrize("input_dims", [([]), ([40, 400]), ([10, 0, 1])])
def test__raise_error_invalid_input_dims(input_dims: list[int]):
    setting = LayerNormSetting()

    with pytest.raises(ValueError):
        _ = setting.gather_input_dims(*input_dims)


@given(st.integers(min_value=1))
@settings(max_examples=100)
def test__nodes_is_update_after_overwrite_nodes(
    n_nodes: int,
):
    setting = LayerNormSetting()

    update_nodes = [n_nodes, n_nodes]
    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


def test__reference_is_not_necessary():
    setting = LayerNormSetting()

    assert not setting.need_reference


# region E2E tests only for MLPSettings

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/layer_norm_setting"


@pytest.mark.parametrize("yaml_file", ["check_layer_norm_nodes.yml"])
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
