import pathlib
from collections.abc import Callable

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import assume, given, settings
from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import EnEquivariantMLPSetting
from phlower.utils.enums import ActivationType


@given(
    nodes=st.lists(
        st.integers(min_value=1, max_value=100),
        min_size=2,
        max_size=5,
    ),
    activation=st.sampled_from(ActivationType),
    dropout=st.floats(width=32, min_value=0, max_value=1.0, exclude_max=True),
    bias=st.booleans(),
    create_linear_weight=st.booleans(),
    norm_function_name=st.sampled_from(ActivationType),
)
def test__can_accept_items(
    nodes: list[int],
    activation: ActivationType,
    dropout: float,
    bias: bool,
    create_linear_weight: bool,
    norm_function_name: ActivationType,
):
    _ = EnEquivariantMLPSetting(
        nodes=nodes,
        activations=[activation.name] * (len(nodes) - 1),
        dropouts=[dropout] * (len(nodes) - 1),
        bias=bias,
        create_linear_weight=create_linear_weight,
        norm_function_name=norm_function_name.name,
    )


def test__default_items():
    setting = EnEquivariantMLPSetting(
        nodes=[-1, 20],
        activations=["identity"],
        dropouts=[],
    )

    assert setting.bias is True
    assert setting.create_linear_weight is False
    assert setting.norm_function_name == ActivationType.identity


@pytest.mark.parametrize(
    "nodes, activations, desired_msg",
    [
        ([10], [], "size of nodes must be larger than 1 "),
        (
            [-1, -1, 10],
            ["identity", "tanh"],
            "nodes in EnEquivariantMLPSetting is inconsistent.",
        ),
    ],
)
def test__raise_error_when_invalid_n_nodes(
    nodes: list[int], activations: list[str], desired_msg: str
):
    with pytest.raises(ValueError) as ex:
        _ = EnEquivariantMLPSetting(
            nodes=nodes,
            activations=activations,
        )
    assert desired_msg in str(ex.value)


@pytest.mark.parametrize(
    "nodes, activations, dropouts",
    [([10, 20, 30], [], []), ([10, 20, 30, 40, 50], [], [])],
)
def test__fill_default_settings(
    nodes: list[int], activations: list[str], dropouts: list[float]
):
    setting = EnEquivariantMLPSetting(
        nodes=nodes, activations=activations, dropouts=dropouts
    )
    desired_activations = ["identity" for _ in range(len(nodes) - 1)]
    desired_dropouts = [0 for _ in range(len(nodes) - 1)]

    assert setting.activations == desired_activations
    assert setting.dropouts == desired_dropouts


@pytest.mark.parametrize(
    "nodes, activations, desired_msg",
    [
        (
            [10, 10, 20],
            ["identity"],
            "Size of nodes and activations is not compatible",
        ),
        (
            [1, 1, 10],
            ["tanh", "tanh", "tanh"],
            "Size of nodes and activations is not compatible",
        ),
    ],
)
def test__raise_error_when_activation_size_is_not_valid(
    nodes: list[int], activations: list[str], desired_msg: str
):
    with pytest.raises(ValueError) as ex:
        _ = EnEquivariantMLPSetting(
            nodes=nodes,
            activations=activations,
        )
    assert desired_msg in str(ex.value)


@pytest.mark.parametrize("input_dims", [([30]), ([40]), ([100])])
def test__gather_input_dims(input_dims: list[int]):
    setting = EnEquivariantMLPSetting(
        nodes=[10, 20],
        activations=["identity"],
        dropouts=[0.1],
    )
    assert setting.gather_input_dims(*input_dims) == input_dims[0]


def test__raise_error_when_invalid_input_dims():
    setting = EnEquivariantMLPSetting(
        nodes=[10, 20],
        activations=["identity"],
        dropouts=[0.1],
    )
    with pytest.raises(ValueError):
        _ = setting.gather_input_dims(40, 400, 10)


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
    setting = EnEquivariantMLPSetting(
        nodes=nodes,
        activations=["identity" for _ in range(len(nodes) - 1)],
        dropouts=[0.1 for _ in range(len(nodes) - 1)],
    )

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


def test__reference_is_not_necessary():
    setting = EnEquivariantMLPSetting(
        nodes=[10, 20], activations=["identity"], dropouts=[0.1]
    )

    assert not setting.need_reference


# region E2E tests only for EnEquivariantMLPSettings

_TEST_DATA_DIR = (
    pathlib.Path(__file__).parent / "data/en_equivariant_mlp_setting"
)


@pytest.mark.parametrize("yaml_file", ["check_en_mlp_nodes.yml"])
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
