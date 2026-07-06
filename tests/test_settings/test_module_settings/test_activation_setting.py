import hypothesis.strategies as st
import pytest
from hypothesis import given

from phlower.settings._module_settings import ActivationSetting


@pytest.mark.parametrize("nodes", [([-1, 20]), ([10, 10])])
def test__can_accept_valid_n_nodes(nodes: list[int]):
    _ = ActivationSetting(nodes=nodes)


@pytest.mark.parametrize("nodes", [([5]), ([10, 10, 10], [], None)])
def test__raise_error_when_invalid_n_nodes(nodes: list[int]):
    with pytest.raises(ValueError):
        _ = ActivationSetting(nodes=nodes)


@pytest.mark.parametrize("input_dims", [[5, 10]])
def test__raise_error_when_multiple_inputs(input_dims: list[int]):
    setting = ActivationSetting(nodes=[-1, 20])

    with pytest.raises(ValueError):
        _ = setting.gather_input_dims(*input_dims)


@pytest.mark.parametrize("input_dims, desired", [([10], 10), ([20], 20)])
def test__gather_input_dims(input_dims: list[int], desired: int):
    setting = ActivationSetting(nodes=[-1, 20])

    assert setting.gather_input_dims(*input_dims) == desired


@given(st.integers(min_value=1))
def test__nodes_is_update_after_overwrite_nodes(node: int):
    update_nodes = [node, node]
    setting = ActivationSetting(nodes=[-1, node])

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


def test__reference_is_not_necessary():
    setting = ActivationSetting(nodes=[-1, 20])
    assert not setting.need_reference
