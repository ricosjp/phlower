import pathlib
from collections.abc import Callable

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import assume, given, settings
from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import DeepSetsSetting
from phlower.utils.enums import ActivationType

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/share_settings"


@pytest.mark.parametrize(
    "nodes, activations, dropouts",
    [
        ([10, 20, 30], ["tanh", "tanh"], [0.2, 0.1]),
        ([10, 30], ["identity"], [0.3]),
        ([5, 10, 20, 5], ["relu", "relu", "tanh"], [0.3, 0.2, 0.1]),
        ([-1, 20, 30], ["tanh", "tanh"], [0.2, 0.1]),
    ],
)
@given(
    last_activation=st.sampled_from(ActivationType),
    pool_operator_name=st.sampled_from(["max", "mean"]),
)
def test__can_accept_valid_n_nodes(
    nodes: list[int],
    activations: list[str],
    dropouts: list[float],
    last_activation: ActivationType,
    pool_operator_name: str,
):
    _ = DeepSetsSetting(
        nodes=nodes,
        activations=activations,
        dropouts=dropouts,
        last_activation=last_activation.name,
        pool_operator=pool_operator_name,
    )


@pytest.mark.parametrize("last_activation", ["aaa", "not_existing"])
def test__raise_error_when_not_registered_activation_name(last_activation: str):
    with pytest.raises(ValueError) as ex:
        _ = DeepSetsSetting(
            nodes=[10, 10],
            pool_operator="mean",
            last_activation=last_activation,
        )
    assert f"{last_activation} is not implemented." in str(ex.value)


@pytest.mark.parametrize(
    "nodes, activations, dropouts",
    [
        ([10, 20, 30], ["identity"], []),
        ([10, 30], [], [0.3, 0.4]),
        ([5, 10, 20, 5], ["relu", "relu", "tanh", "identity"], [0.3, 0.2, 0.1]),
        ([5, -1, 20, 5], ["relu", "relu", "tanh"], [0.3, 0.2, 0.1]),
        ([5, 10, 20, 5], ["relu", "relu", "tanh"], [0.3]),
        ([10], [], []),
    ],
)
def test__raise_error_when_invalid_n_nodes(
    nodes: list[int], activations: list[str], dropouts: list[float]
):
    with pytest.raises(ValueError):
        _ = DeepSetsSetting(
            nodes=nodes,
            activations=activations,
            dropouts=dropouts,
            pool_operator="mean",
            last_activation="relu",
        )


@pytest.mark.parametrize(
    "nodes, activations, dropouts",
    [([10, 20, 30], [], []), ([10, 20, 30, 40, 50], [], [])],
)
def test__fill_default_settings(
    nodes: list[int], activations: list[str], dropouts: list[float]
):
    setting = DeepSetsSetting(
        nodes=nodes, activations=activations, dropouts=dropouts
    )
    desired_activations = ["identity" for _ in range(len(nodes) - 1)]
    desired_dropouts = [0 for _ in range(len(nodes) - 1)]

    assert setting.activations == desired_activations
    assert setting.dropouts == desired_dropouts
    assert setting.bias
    assert setting.pool_operator == "max"
    assert setting.last_activation == "identity"


@pytest.mark.parametrize("input_dims", [([30]), ([40]), ([100])])
def test__gather_input_dims(input_dims: list[int]):
    setting = DeepSetsSetting(
        nodes=[10, 20], activations=["identity"], dropouts=[0.1]
    )

    assert setting.gather_input_dims(*input_dims) == input_dims[0]


@pytest.mark.parametrize("input_dims", [([]), ([40, 400]), ([10, 0, 1])])
def test__raise_error_invalid_input_dims(input_dims: list[int]):
    setting = DeepSetsSetting(
        nodes=[10, 20], activations=["identity"], dropouts=[0.1]
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
    setting = DeepSetsSetting(
        nodes=nodes,
        activations=["identity" for _ in range(len(nodes) - 1)],
        dropouts=[0.1 for _ in range(len(nodes) - 1)],
    )

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


def test__reference_is_not_necessary():
    setting = DeepSetsSetting(
        nodes=[10, 20], activations=["identity"], dropouts=[0.1]
    )

    assert not setting.need_reference


# region E2E tests only for MLPSettings

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/deepsets_setting"


@pytest.mark.parametrize("yaml_file", ["check_deepsets_nodes.yml"])
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
