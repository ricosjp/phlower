import pathlib
from collections.abc import Callable

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import assume, given, settings

from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import ReducerSetting

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/share_settings"


@pytest.mark.parametrize(
    "nodes, activation, operator",
    [
        ([10, 20, 30], "tanh", "add"),
        ([10, 30], "identity", "add"),
        ([5, 10, 20, 5], "relu", "mul"),
        ([-1, 20, 30], "tanh", "mul"),
    ],
)
def test__can_accept_valid_n_nodes(
    nodes: list[int], activation: str, operator: str
):
    _ = ReducerSetting(nodes=nodes, activation=activation, operator=operator)


@pytest.mark.parametrize(
    "nodes, activation",
    [
        ([10, 20, 30], "identity"),
        ([10, 30], "identity"),
    ],
)
def test__raise_error_when_invalid_n_nodes(nodes: list[int], activation: str):
    with pytest.raises(ValueError):
        _ = ReducerSetting(nodes=nodes, activation=activation)


@pytest.mark.parametrize(
    "nodes, activation, operator",
    [
        ([10, 20, 30], "identity", "add"),
        ([10, 20, 30, 40, 50], "identity", "mul"),
    ],
)
def test__fill_default_settings(
    nodes: list[int], activation: str, operator: str
):
    setting = ReducerSetting(
        nodes=nodes, activation=activation, operator=operator
    )
    desired_activation = "identity"
    desired_operator = operator

    assert setting.activation == desired_activation
    assert setting.operator == desired_operator


@pytest.mark.parametrize("input_dims", [([30]), ([40]), ([100])])
def test__gather_input_dims(input_dims: list[int]):
    setting = ReducerSetting(
        nodes=[10, 20], activation="identity", operator="add"
    )

    assert setting.gather_input_dims(*input_dims) == input_dims[0]


@pytest.mark.parametrize("input_dims", [([]), ([40, 400]), ([10, 0, 1])])
def test__raise_error_invalid_input_dims(input_dims: list[int]):
    setting = ReducerSetting(
        nodes=[10, 20], activation="identity", operator="add"
    )

    with pytest.raises(ValueError):
        _ = setting.gather_input_dims(*input_dims)


@st.composite
def same_length_lists(draw: Callable) -> tuple[list[int], list[int]]:
    n_elements = draw(st.integers(min_value=2, max_value=10))
    fixed_length_list = st.lists(
        st.integers(min_value=1, max_value=200),
        min_size=n_elements,
        max_size=n_elements,
    )

    return (draw(fixed_length_list), draw(fixed_length_list))


@given(same_length_lists())
@settings(max_examples=100)
def test__nodes_is_update_after_overwrite_nodes(
    lists: tuple[list[int], list[int]],
):
    nodes, update_nodes = lists
    assume(nodes != update_nodes)
    setting = ReducerSetting(
        nodes=nodes,
        activation="identity",
        operator="add",
    )

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


def test__reference_is_not_necessary():
    setting = ReducerSetting(
        nodes=[10, 20], activation="identity", operator="add"
    )

    assert not setting.need_reference


# region E2E tests only for ReducerSettings

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/reducer_setting"


@pytest.mark.parametrize("yaml_file", ["check_reducer_nodes.yml"])
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
