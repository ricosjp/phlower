import pathlib
from collections.abc import Callable

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import given
from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import ContractionSetting


@pytest.mark.parametrize("nodes", [([-1, 20]), ([10, 10])])
def test__can_accept_valid_n_nodes(nodes: list[int]):
    _ = ContractionSetting(nodes=nodes)


@pytest.mark.parametrize(
    "nodes, desired_msg",
    [
        ([-1, -1], "Nodes in ContractionSetting are inconsistent."),
        ([10, 10, 10], "Length of nodes must be 2."),
    ],
)
def test__raise_error_when_invalid_n_nodes(nodes: list[int], desired_msg: str):
    with pytest.raises(ValueError) as ex:
        _ = ContractionSetting(nodes=nodes)

    assert desired_msg in str(ex.value)


@pytest.mark.parametrize(
    "input_dims, desired", [([30, 50], 80), ([40], 40), ([100, 10], 110)]
)
def test__gather_input_dims(input_dims: list[int], desired: int):
    setting = ContractionSetting(nodes=[-1, 20])

    assert setting.gather_input_dims(*input_dims) == desired


@st.composite
def same_length_lists(draw: Callable) -> tuple[list[int]]:
    n_elements = draw(st.integers(min_value=2, max_value=2))
    fixed_length_list = st.lists(
        st.integers(min_value=1, max_value=200),
        min_size=n_elements,
        max_size=n_elements,
    )

    return (draw(fixed_length_list), draw(fixed_length_list))


@given(st.integers(min_value=1), st.integers(min_value=1))
def test__nodes_is_update_after_overwrite_nodes(node1: int, node2: int):
    update_nodes = [node1, node2]
    setting = ContractionSetting(nodes=[-1, node2])

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


@pytest.mark.parametrize(
    "nodes, update_nodes, desired_msg",
    [
        ([-1, 20], [20, 30], "the last value of nodes is not consistent."),
        ([-1, 20], [10, 10, 20], "Invalid length of nodes to overwrite."),
        ([10, 20], [-1, 20], "Resolved nodes must be positive."),
    ],
)
def test__invalid_update_nodes(
    nodes: list[int], update_nodes: list[int], desired_msg: str
):
    setting = ContractionSetting(nodes=nodes)

    with pytest.raises(ValueError) as ex:
        setting.overwrite_nodes(update_nodes)

    assert desired_msg in str(ex.value)


def test__reference_is_not_necessary():
    setting = ContractionSetting(nodes=[-1, 20])

    assert not setting.need_reference


# region E2E tests only for MLPSettings

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/contraction_setting"


@pytest.mark.parametrize("yaml_file", ["check_contraction_nodes.yml"])
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
