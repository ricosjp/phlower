import pathlib
from collections.abc import Callable

import hypothesis.strategies as st
import pydantic
import pytest
import yaml
from hypothesis import assume, given, settings

from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import SlipSetting


@pytest.mark.parametrize("nodes", [(None), ([10, 10])])
def test__can_accept_valid_n_nodes(nodes: list[int] | None):
    _ = SlipSetting(nodes=nodes, normal_name="", flag_name="")


@pytest.mark.parametrize("nodes", [([5]), ([10, 10, 10])])
def test__raise_error_when_invalid_n_nodes(nodes: list[int]):
    with pytest.raises(pydantic.ValidationError) as ex:
        _ = SlipSetting(nodes=nodes, normal_name="", flag_name="")

    assert "Size of nodes must be 2" in str(ex.value)


@pytest.mark.parametrize(
    "input_dims, desired",
    [([30, 1, 1], 32), ([40, 1, 1], 42), ([100, 3, 1], 104)],
)
def test__gather_input_dims(input_dims: list[int], desired: int):
    setting = SlipSetting(normal_name="", flag_name="")

    assert setting.gather_input_dims(*input_dims) == desired


@pytest.mark.parametrize("input_dims", [([30]), ([30, 1]), ([30, 1, 1, 1])])
def test__raise_error_when_invalid_input_dims(input_dims: list[int]):
    setting = SlipSetting(normal_name="", flag_name="")

    with pytest.raises(ValueError) as ex:
        setting.gather_input_dims(*input_dims)

    assert "num of input should be 3" in str(ex.value)


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
    setting = SlipSetting(normal_name="", flag_name="")

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


def test__reference_is_not_necessary():
    setting = SlipSetting(normal_name="", flag_name="")

    assert not setting.need_reference


# region E2E tests only for MLPSettings

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/slip_setting"


@pytest.mark.parametrize("yaml_file", ["check_slip_nodes.yml"])
def test__nodes_after_resolve(yaml_file: str):
    with open(_TEST_DATA_DIR / yaml_file) as fr:
        content = yaml.load(fr, Loader=yaml.SafeLoader)

    setting = PhlowerModelSetting(**content["model"])
    setting.network.resolve(is_first=True)

    assert len(content["misc"]["tests"].items()) > 0

    for key, value in content["misc"]["tests"].items():
        target = setting.network.search_module_setting(key)
        assert target.get_n_nodes() == value


@pytest.mark.parametrize(
    "yaml_file, missing_name",
    [
        ("not_found_normal_name.yml", "mlp1_missing"),
        ("not_found_flag_name.yml", "mlp2_missing"),
    ],
)
def test__raise_error_when_input_name_is_missing(
    yaml_file: str, missing_name: str
):
    with open(_TEST_DATA_DIR / yaml_file) as fr:
        content = yaml.load(fr, Loader=yaml.SafeLoader)

    setting = PhlowerModelSetting(**content["model"])

    with pytest.raises(ValueError) as ex:
        setting.network.resolve(is_first=True)
    assert f"{missing_name} is not found in input_keys" in str(ex.value)


# endregion
