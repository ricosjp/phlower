import pathlib

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import given

from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import SPMMSetting


@pytest.mark.parametrize("nodes", [[-1, 20], [10, 10], None])
def test__can_accept_valid_n_nodes(nodes: list[int] | None):
    _ = SPMMSetting(nodes=nodes, support_name="dummy")


@pytest.mark.parametrize(
    "nodes, desired_msg",
    [
        ([5], "length of nodes must be 2."),
        ([10, 10, 10], "length of nodes must be 2."),
        ([], "length of nodes must be 2."),
        ([1, -1], "Nodes in SPMM is inconsistent."),
    ],
)
def test__raise_error_when_invalid_n_nodes(nodes: list[int], desired_msg: str):
    with pytest.raises(ValueError) as ex:
        _ = SPMMSetting(nodes=nodes, support_name="dummy")
    assert desired_msg in str(ex.value)


@pytest.mark.parametrize(
    "input_dims, desired", [([30], 30), ([40], 40), ([100], 100)]
)
def test__gather_input_dims(input_dims: list[int], desired: int):
    setting = SPMMSetting(support_name="dummy")

    assert setting.gather_input_dims(*input_dims) == desired


def test__reference_is_not_necessary():
    setting = SPMMSetting(support_name="dummy")
    assert not setting.need_reference


@given(st.integers(min_value=1), st.integers(min_value=1))
def test__nodes_is_update_after_overwrite_nodes(node1: int, node2: int):
    update_nodes = [node1, node2]
    setting = SPMMSetting(support_name="dummy", nodes=[-1, node2])

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


# region E2E tests only for SPMMSettings

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/spmm_setting"


@pytest.mark.parametrize(
    "yaml_file",
    ["check_spmm_nodes.yml", "check_spmm_nodes_2.yml"],
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
