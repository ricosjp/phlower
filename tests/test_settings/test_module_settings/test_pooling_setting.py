import pathlib
from collections.abc import Callable

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import assume, given, settings
from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import PoolingSetting


@pytest.mark.parametrize("nodes", [(None), ([10, 10])])
def test__can_accept_valid_n_nodes(nodes: list[int] | None):
    _ = PoolingSetting(nodes=nodes)


@pytest.mark.parametrize("nodes", [([5]), ([10, 10, 10])])
def test__raise_error_when_invalid_n_nodes(nodes: list[int]):
    with pytest.raises(ValueError) as ex:
        _ = PoolingSetting(nodes=nodes)
    assert "size of nodes must be 2 in PoolingSetting" in str(ex.value)


@pytest.mark.parametrize(
    "input_dims, desired", [([30], 30), ([40], 40), ([100], 100)]
)
def test__gather_input_dims(input_dims: list[int], desired: int):
    setting = PoolingSetting()

    assert setting.gather_input_dims(*input_dims) == desired


def test__reference_is_not_necessary():
    setting = PoolingSetting()

    assert not setting.need_reference


@st.composite
def same_length_lists(draw: Callable) -> tuple[list[int]]:
    n_elements = draw(st.integers(min_value=2, max_value=2))
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
    setting = PoolingSetting()

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


@pytest.mark.parametrize("pooling_name", ["mymax", "sum"])
def raise_error_when_invalid_pooling_type(pooling_name: str):
    with pytest.raises(ValueError) as ex:
        _ = PoolingSetting(pool_operator_name=pooling_name)

    assert f"Pooling operator {pooling_name} is not implemented." in str(
        ex.value
    )


# region E2E tests only for PoolingSettings

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/pooling_setting"


@pytest.mark.parametrize("yaml_file", ["check_pooling_nodes.yml"])
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
