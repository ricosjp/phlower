import pathlib
from collections.abc import Callable

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import assume, given, settings
from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import EnEquivariantTCNSetting
from phlower.utils.enums import ActivationType


@st.composite
def create_tcn_setting_items(
    draw: Callable,
) -> tuple[list[int], list[int], list[int], list[str], list[float], bool]:
    n_nodes = draw(st.integers(min_value=2, max_value=5))
    nodes = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=n_nodes,
            max_size=n_nodes,
        )
    )

    activation = draw(st.sampled_from(ActivationType))
    n_items = n_nodes - 1
    kernel_sizes = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=n_items,
            max_size=n_items,
        )
    )
    dilations = draw(
        st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=n_items,
            max_size=n_items,
        )
    )
    activations = [activation.name for _ in range(n_items)]
    dropouts = draw(
        st.lists(
            st.floats(width=32, min_value=0, max_value=1.0, exclude_max=True),
            min_size=n_items,
            max_size=n_items,
        )
    )
    bias = draw(st.booleans())

    return nodes, kernel_sizes, dilations, activations, dropouts, bias


@given(items=create_tcn_setting_items())
def test__can_accept_valid_n_nodes(items: tuple):
    nodes, kernel_sizes, dilations, activations, dropouts, bias = items
    _ = EnEquivariantTCNSetting(
        nodes=nodes,
        kernel_sizes=kernel_sizes,
        activations=activations,
        dropouts=dropouts,
        bias=bias,
        dilations=dilations,
    )


@pytest.mark.parametrize(
    "nodes, kernel_sizes, activations, dropouts, dilations, desired_msg",
    [
        (
            [10, 20, 30],
            [3, 3],
            ["identity"],
            [],
            [],
            "Size of nodes and activations is not compatible",
        ),
        (
            [10, 30],
            [2],
            [],
            [0.3, 0.4],
            [],
            "Size of nodes and dropouts is not compatible",
        ),
        (
            [5, 10, 20, 5],
            [3, 3, 3],
            ["relu", "relu", "tanh", "identity"],
            [0.3, 0.2, 0.1],
            [],
            "Size of nodes and activations is not compatible",
        ),
        (
            [5, -1, 20, 5],
            [3, 1, 3],
            ["relu", "relu", "tanh"],
            [0.3, 0.2, 0.1],
            [],
            "nodes in EnEquivariantTCN is inconsistent.",
        ),
        (
            [5, 10, 20, 5],
            [3, 5, 5],
            ["relu", "relu", "tanh"],
            [0.3],
            [1, 2, 3],
            "Size of nodes and dropouts is not compatible",
        ),
        ([10], [], [], [], [], "size of nodes must be larger than 1"),
        (
            [10, 30, 40],
            [4, 4, 4],
            [],
            [],
            [],
            "Size of nodes and kernel_sizes is not compatible",
        ),
        (
            [10, 30, 40],
            [4, 4],
            [],
            [],
            [1, 2, 5],
            "Size of nodes and dilations is not compatible",
        ),
    ],
)
def test__raise_error_when_invalid_n_nodes(
    nodes: list[int],
    kernel_sizes: list[int],
    activations: list[str],
    dropouts: list[float],
    dilations: list[int],
    desired_msg: str,
):
    with pytest.raises(ValueError) as ex:
        _ = EnEquivariantTCNSetting(
            nodes=nodes,
            kernel_sizes=kernel_sizes,
            activations=activations,
            dropouts=dropouts,
            dilations=dilations,
        )
    assert desired_msg in str(ex.value)


@pytest.mark.parametrize("nodes", [[5, 5, 10], [10, 20, 3]])
def test__raise_error_for_linear_weight_flag(nodes: list[int]):
    setting = EnEquivariantTCNSetting(
        nodes=nodes,
        kernel_sizes=[2 for _ in range(len(nodes) - 1)],
        create_linear_weight=False,
    )

    with pytest.raises(ValueError) as ex:
        _ = setting.confirm(None)

    assert "create_linear_weight must be True" in str(ex.value)


@pytest.mark.parametrize(
    "nodes, kernel_sizes",
    [([10, 20, 30], [3, 4]), ([10, 20, 30, 40, 50], [2, 2, 2, 2])],
)
def test__fill_default_settings(nodes: list[int], kernel_sizes: list[str]):
    setting = EnEquivariantTCNSetting(nodes=nodes, kernel_sizes=kernel_sizes)
    desired_activations = ["identity" for _ in range(len(nodes) - 1)]
    desired_dropouts = [0 for _ in range(len(nodes) - 1)]
    desired_dilations = [1 for _ in range(len(nodes) - 1)]

    assert setting.activations == desired_activations
    assert setting.dropouts == desired_dropouts
    assert setting.bias
    assert setting.dilations == desired_dilations


@pytest.mark.parametrize("input_dims", [([30]), ([40]), ([100])])
def test__gather_input_dims(input_dims: list[int]):
    setting = EnEquivariantTCNSetting(nodes=[10, 20], kernel_sizes=[2])

    assert setting.gather_input_dims(*input_dims) == input_dims[0]


@pytest.mark.parametrize("input_dims", [([]), ([40, 400]), ([10, 0, 1])])
def test__raise_error_invalid_input_dims(input_dims: list[int]):
    setting = EnEquivariantTCNSetting(nodes=[10, 20], kernel_sizes=[3])

    with pytest.raises(ValueError) as ex:
        _ = setting.gather_input_dims(*input_dims)

    assert "only one input is allowed" in str(ex.value)


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
    setting = EnEquivariantTCNSetting(
        nodes=nodes, kernel_sizes=[3 for _ in range(len(nodes) - 1)]
    )

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


def test__reference_is_not_necessary():
    setting = EnEquivariantTCNSetting(nodes=[10, 20], kernel_sizes=[3])

    assert not setting.need_reference


# region E2E tests only for MLPSettings

_TEST_DATA_DIR = (
    pathlib.Path(__file__).parent / "data/en_equivariant_tcn_setting"
)


@pytest.mark.parametrize("yaml_file", ["check_en_tcn_nodes.yml"])
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
