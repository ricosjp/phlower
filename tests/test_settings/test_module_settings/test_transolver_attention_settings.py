import pathlib

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import assume, given
from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import TransolverAttentionSetting


@pytest.mark.parametrize(
    "nodes, heads",
    [
        ([10, 20, 30], 10),
        ([-1, 30, 20], 10),
        ([16, 32, 16], 8),
        ([16, 16, 16], 4),
    ],
)
def test__can_accept_valid_n_nodes(nodes: list[int], heads: int):
    _ = TransolverAttentionSetting(nodes=nodes, heads=heads)


@pytest.mark.parametrize(
    "nodes, heads",
    [
        ([10], 2),
        ([10, 10], 2),
        ([10, 10, 10, 10], 2),
        ([30, -20, 30], 10),
        ([-1, -1, -1], 8),
    ],
)
def test__raise_error_when_invalid_n_nodes(nodes: list[int], heads: int):
    with pytest.raises(ValueError):
        _ = TransolverAttentionSetting(nodes=nodes, heads=heads)


@pytest.mark.parametrize(
    "heads",
    [1, 2, 4, 8, 16, 32],
)
def test__can_accept_valid_heads(heads: int):
    _ = TransolverAttentionSetting(nodes=[16, heads * 2, 16], heads=heads)


@pytest.mark.parametrize(
    "heads",
    [0, -1, -10],
)
def test__raise_error_when_invalid_heads(heads: int):
    with pytest.raises(ValueError):
        _ = TransolverAttentionSetting(nodes=[16, 16, 16], heads=heads)


@pytest.mark.parametrize(
    "slice_num",
    [1, 10, 32, 64, 100],
)
def test__can_accept_valid_slice_num(slice_num: int):
    _ = TransolverAttentionSetting(
        nodes=[16, 16, 16], heads=8, slice_num=slice_num
    )


@pytest.mark.parametrize(
    "slice_num",
    [0, -1, -32],
)
def test__raise_error_when_invalid_slice_num(slice_num: int):
    with pytest.raises(ValueError):
        _ = TransolverAttentionSetting(
            nodes=[16, 16, 16], heads=8, slice_num=slice_num
        )


@pytest.mark.parametrize(
    "dropout",
    [0.0, 0.1, 0.5, 0.9, 0.99],
)
def test__can_accept_valid_dropout(dropout: float):
    _ = TransolverAttentionSetting(nodes=[16, 16, 16], heads=8, dropout=dropout)


@pytest.mark.parametrize(
    "dropout",
    [-0.1, 1.0, 1.5, 2.0],
)
def test__raise_error_when_invalid_dropout(dropout: float):
    with pytest.raises(ValueError):
        _ = TransolverAttentionSetting(
            nodes=[16, 16, 16], heads=8, dropout=dropout
        )


@pytest.mark.parametrize(
    "nodes, heads",
    [
        ([16, 17, 16], 8),
        ([16, 15, 16], 8),
        ([10, 21, 10], 4),
    ],
)
def test__raise_error_when_nodes_not_divisible_by_heads(
    nodes: list[int], heads: int
):
    with pytest.raises(ValueError):
        _ = TransolverAttentionSetting(nodes=nodes, heads=heads)


@pytest.mark.parametrize("input_dims", [([30]), ([40]), ([100])])
def test__gather_input_dims(input_dims: list[int]):
    setting = TransolverAttentionSetting(nodes=[16, 16, 16], heads=8)

    assert setting.gather_input_dims(*input_dims) == input_dims[0]


@pytest.mark.parametrize("input_dims", [([]), ([40, 400]), ([10, 0, 1])])
def test__raise_error_invalid_input_dims(input_dims: list[int]):
    setting = TransolverAttentionSetting(nodes=[16, 16, 16], heads=8)

    with pytest.raises(ValueError):
        _ = setting.gather_input_dims(*input_dims)


@given(
    nodes_factor=st.lists(
        st.integers(min_value=1, max_value=200), min_size=3, max_size=3
    ),
    udpate_nodes_factor=st.lists(
        st.integers(min_value=1, max_value=200), min_size=3, max_size=3
    ),
)
def test__nodes_is_update_after_overwrite_nodes(
    nodes_factor: list[int], udpate_nodes_factor: list[int]
):
    assume(nodes_factor != udpate_nodes_factor)
    heads = 8
    nodes = [x * heads for x in nodes_factor]
    update_nodes = [x * heads for x in udpate_nodes_factor]

    setting = TransolverAttentionSetting(nodes=nodes, heads=heads)

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


def test__reference_is_not_necessary():
    setting = TransolverAttentionSetting(nodes=[16, 16, 16], heads=8)

    assert not setting.need_reference


# region E2E tests only for TransolverAttentionSettings

_TEST_DATA_DIR = (
    pathlib.Path(__file__).parent / "data/transolver_attention_setting"
)


@pytest.mark.parametrize("yaml_file", ["check_transolver_attention_nodes.yml"])
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
