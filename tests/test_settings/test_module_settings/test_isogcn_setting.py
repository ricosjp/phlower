import pathlib
from collections.abc import Callable

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import assume, given, settings
from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import IsoGCNSetting


def test__use_flag_when_default_setting():
    _setting = IsoGCNSetting(isoam_names=["dummmy"])

    assert not _setting.self_network.use_network
    assert not _setting.coefficient_network.use_network
    assert not _setting.neumann_setting.use_neumann


@pytest.mark.parametrize(
    "isoam_names", [([]), (["a", "b", "c", "d"]), (["a", "b", "c", "d", "e"])]
)
def test__check_length_of_isoam_names(isoam_names: list[str]):
    with pytest.raises(ValueError):
        _ = IsoGCNSetting(isoam_names=isoam_names)


@pytest.mark.parametrize(
    "nodes, activations, dropouts",
    [
        ([10, 20, 30], ["tanh"], [0.2]),
        ([10, 30], ["identity"], [0.3]),
        ([5, 10, 20, 5], ["relu"], [0.3]),
        ([-1, 20, 30], ["tanh"], [0.1]),
        ([5, 10, 20, 5], [], []),
    ],
)
def test__can_accept_valid_parameters_for_self_network(
    nodes: list[int], activations: list[str], dropouts: list[float]
):
    _ = IsoGCNSetting(
        nodes=nodes,
        isoam_names=["dummy"],
        self_network={
            "activations": activations,
            "dropouts": dropouts,
            "bias": True,
        },
    )


@pytest.mark.parametrize(
    "nodes, activations, dropouts",
    [
        ([10, 20, 30], ["tanh", "tanh"], [0.2]),
        ([10, 30], ["identity", "identity"], []),
        ([5, 10, 20, 5], ["relu", "relu", "relu"], [0.3, 0.4, 0.0]),
        ([-1, 20, 30], [], [0.1, 0.1]),
    ],
)
def test__raise_error_invalid_parameters_for_self_network(
    nodes: list[int], activations: list[str], dropouts: list[float]
):
    with pytest.raises(ValueError):
        _ = IsoGCNSetting(
            nodes=nodes,
            isoam_names=["dummy"],
            self_network={
                "activations": activations,
                "dropouts": dropouts,
                "bias": True,
            },
        )


@pytest.mark.parametrize(
    "nodes, activations, dropouts",
    [
        ([10, 20, 30], ["tanh", "tanh"], [0.2, 0.1]),
        ([10, 30], ["identity"], [0.3]),
        ([5, 10, 20, 5], ["relu", "relu", "tanh"], [0.3, 0.2, 0.1]),
        ([-1, 20, 30], ["tanh", "tanh"], [0.2, 0.1]),
    ],
)
def test__can_accept_valid_parameters_for_coefficient_network(
    nodes: list[int], activations: list[str], dropouts: list[float]
):
    _ = IsoGCNSetting(
        nodes=nodes,
        isoam_names=["dummy"],
        coefficient_network={
            "activations": activations,
            "dropouts": dropouts,
            "bias": True,
        },
    )


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
def test__raise_error_invalid_parameters_for_coefficient_network(
    nodes: list[int], activations: list[str], dropouts: list[float]
):
    with pytest.raises(ValueError):
        _ = IsoGCNSetting(
            nodes=nodes,
            isoam_names=["dummy"],
            coefficient_network={
                "activations": activations,
                "dropouts": dropouts,
                "bias": True,
            },
        )


# region test for Neumann setting


def test__raise_error_when_use_neumannn_without_self_network():
    with pytest.raises(ValueError):
        _ = IsoGCNSetting(
            isoam_names=["dummy"], neumann_setting={"factor": 0.2}
        )


@pytest.mark.parametrize(
    "inversed_moment_name, neumann_input_name",
    [(None, None), ("aaa", None), (None, "aaa")],
)
def test__invalid_neumann_setting(
    inversed_moment_name: str, neumann_input_name: str
):
    with pytest.raises(ValueError):
        _ = IsoGCNSetting(
            nodes=[10, 10],
            self_network={"use_network": True},
            isoam_names=["dummy"],
            neumann_setting={
                "factor": 0.2,
                "inversed_moment_name": inversed_moment_name,
                "neumann_input_name": neumann_input_name,
            },
        )


# endregion


# region test for resolve


@pytest.mark.parametrize(
    "input_dims, desired",
    [([30], 30), ([40], 40), ([100], 100), ([20, 20], 20)],
)
def test__gather_input_dims(input_dims: list[int], desired: int):
    setting = IsoGCNSetting(
        nodes=[10, 20],
        isoam_names=["dummy"],
    )

    assert setting.gather_input_dims(*input_dims) == desired


@pytest.mark.parametrize("input_dims", [([]), ([40, 400, 10]), ([10, 0, 1])])
def test__raise_error_invalid_input_dims(input_dims: list[int]):
    setting = IsoGCNSetting(
        nodes=[10, 20],
        isoam_names=["dummy"],
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
    setting = IsoGCNSetting(
        nodes=nodes,
        isoam_names=["dummy"],
    )

    before_nodes = setting.get_n_nodes()
    assert before_nodes != update_nodes

    setting.overwrite_nodes(update_nodes)
    assert setting.get_n_nodes() == update_nodes


def test__reference_is_not_necessary():
    setting = IsoGCNSetting(nodes=[10, 20], isoam_names=["dummy"])

    assert not setting.need_reference


# endregion


# region E2E tests

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/isogcn_setting"


@pytest.mark.parametrize("yaml_file", ["check_isogcn_nodes.yml"])
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
