import pathlib

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import given
from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import EinsumSetting


@pytest.mark.parametrize("nodes", [([-1, 20]), ([10, 10])])
def test__can_accept_valid_n_nodes(nodes: list[int]):
    _ = EinsumSetting(nodes=nodes, equation="")


@pytest.mark.parametrize("nodes", [([5]), ([10, 10, 10], [], None)])
def test__raise_error_when_invalid_n_nodes(nodes: list[int]):
    with pytest.raises(ValueError):
        _ = EinsumSetting(nodes=nodes)


@pytest.mark.parametrize(
    "input_dims, desired", [([30, 50, 40], 120), ([40], 40), ([100, 10], 110)]
)
def test__gather_input_dims(input_dims: list[int], desired: int):
    setting = EinsumSetting(equation="", nodes=[-1, 20])

    assert setting.gather_input_dims(*input_dims) == desired


@given(st.integers(min_value=1), st.integers(min_value=1))
def test__nodes_is_update_after_overwrite_nodes(node1: int, node2: int):
    update_nodes = [node1, node2]
    setting = EinsumSetting(equation="", nodes=[-1, node2])

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


def test__reference_is_not_necessary():
    setting = EinsumSetting(equation="", nodes=[-1, 20])
    assert not setting.need_reference


# region E2E tests only for MLPSettings

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/einsum_setting"


@pytest.mark.parametrize(
    "yaml_file",
    ["check_einsum_nodes.yml", "check_einsum_nodes_omit_input_keys.yml"],
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


@pytest.mark.parametrize(
    "yaml_file, desired_msg",
    [
        (
            "invalid_input_orders.yml",
            "mlp1_missing is not defined in input_keys",
        ),
        (
            "invalid_toomany_input_orders.yml",
            "the number of nodes isn't equal to that of input_orders.",
        ),
    ],
)
def test__raise_error_when_invalid_input_orders(
    yaml_file: str, desired_msg: str
):
    with open(_TEST_DATA_DIR / yaml_file) as fr:
        content = yaml.load(fr, Loader=yaml.SafeLoader)

    setting = PhlowerModelSetting(**content["model"])

    with pytest.raises(ValueError) as ex:
        setting.network.resolve(is_first=True)

    assert desired_msg in str(ex.value)


# endregion
