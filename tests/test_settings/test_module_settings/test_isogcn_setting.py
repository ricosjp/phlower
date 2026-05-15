import pathlib
from collections.abc import Callable
from unittest import mock

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


def test__bias_when_default_setting():
    _setting = IsoGCNSetting(
        isoam_names=["dummy"],
        coefficient_network={"use_network": True},
        self_network={"use_network": True},
    )

    assert _setting.self_network.bias is False
    assert _setting.coefficient_network.bias is True


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
    "nodes, activations, dropouts, desired_msg",
    [
        (
            [10, 20, 30],
            ["identity"],
            [],
            "Size of nodes and activations is not compatible",
        ),
        (
            [10, 30],
            [],
            [0.3, 0.4],
            "Size of nodes and dropouts is not compatible",
        ),
        (
            [5, 10, 20, 5],
            ["relu", "relu", "tanh", "identity"],
            [0.3, 0.2, 0.1],
            "Size of nodes and activations is not compatible",
        ),
        (
            [5, -1, 20, 5],
            ["relu", "relu", "tanh"],
            [0.3, 0.2, 0.1],
            "value -1 in 1-th of nodes is not allowed.",
        ),
        (
            [5, 10, 20, 5],
            ["relu", "relu", "tanh"],
            [0.3],
            "Size of nodes and dropouts is not compatible",
        ),
    ],
)
def test__raise_error_invalid_parameters_for_coefficient_network(
    nodes: list[int],
    activations: list[str],
    dropouts: list[float],
    desired_msg: str,
):
    with pytest.raises(ValueError, match=desired_msg):
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
    "inversed_moment_name, neumann_input_name, desired_msg",
    [
        (None, None, "inversed_moment_name must be determined"),
        ("aaa", None, "neumann_input_name must be determined"),
        (None, "aaa", "inversed_moment_name must be determined"),
    ],
)
def test__invalid_neumann_setting(
    inversed_moment_name: str, neumann_input_name: str, desired_msg: str
):
    with pytest.raises(ValueError, match=desired_msg):
        _ = IsoGCNSetting(
            nodes=[10, 10],
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
    [
        ({"x": 30}, 30),
        ({"x": 40}, 40),
        ({"x": 100}, 100),
        ({"x": 20}, 20),
    ],
)
def test__gather_input_dims(input_dims: dict[str, int], desired: int):
    setting = IsoGCNSetting(
        nodes=[10, 20],
        isoam_names=["dummy"],
        self_network={"use_network": True},
    )

    assert setting.gather_input_dims(input_dims) == desired


@pytest.mark.parametrize(
    "input_dims, inversed_moment_source, desired",
    [
        ({"x": 30, "minv": 50, "neumann_input": 40}, "input_data", 30),
        ({"neumann_input": 30, "minv": 50, "x": 100}, "input_data", 100),
        ({"neumann_input": 50, "x": 40}, "field_data", 40),
    ],
)
def test__gather_input_dims_with_neumann(
    input_dims: dict[str, int], inversed_moment_source: str, desired: int
):
    setting = IsoGCNSetting(
        nodes=[10, 20],
        isoam_names=["dummy"],
        neumann_setting={
            "use_neumann": True,
            "inversed_moment_name": "minv",
            "neumann_input_name": "neumann_input",
            "inversed_moment_source": inversed_moment_source,
        },
        self_network={"use_network": True},
    )

    assert setting.gather_input_dims(input_dims) == desired


@pytest.mark.parametrize(
    "use_neumann, inversed_moment_source, input_dims ,desired_n_inputs",
    [
        (False, "input_data", {}, 1),
        (True, "field_data", {"x": 10, "neumann_input": 3, "minv": 4}, 2),
        (True, "field_data", {"x": 10}, 2),
        (True, "input_data", {"x": 10, "neumann_input": 3}, 3),
    ],
)
def test__raise_error_invalid_input_dims(
    use_neumann: bool,
    inversed_moment_source: str | None,
    input_dims: list[int],
    desired_n_inputs: int,
):
    setting = IsoGCNSetting(
        nodes=[10, 20],
        isoam_names=["dummy"],
        neumann_setting={
            "use_neumann": use_neumann,
            "inversed_moment_source": inversed_moment_source,
            "inversed_moment_name": "moment_input",
            "neumann_input_name": "neumann_input",
        },
        self_network={
            "use_network": True,
            "activations": ["tanh"],
            "dropouts": [0.2],
            "bias": True,
        },
    )

    with pytest.raises(
        ValueError,
        match=f"{desired_n_inputs} inputs are necessary "
        "in IsoGCN for this setting",
    ):
        _ = setting.gather_input_dims(input_dims)


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


def test__allow_use_inversed_moment_from_input_data():
    setting = IsoGCNSetting(
        nodes=[10, 20],
        isoam_names=["dummy"],
        self_network={"use_network": True},
        neumann_setting={
            "use_neumann": True,
            "factor": 0.2,
            "inversed_moment_source": "input_data",
            "inversed_moment_name": "moment_input",
            "neumann_input_name": "neumann_input",
        },
    )

    assert setting.neumann_setting.inversed_moment_source == "input_data"

    self_module = mock.MagicMock()
    self_module.get_input_keys.return_value = [
        "x",
        "neumann_input",
        "moment_input",
    ]
    setting.confirm(self_module)
    n_gather_input_dims = setting.gather_input_dims(
        {"neumann_input": 40, "moment_input": 50, "x": 30}
    )
    assert n_gather_input_dims == 30


def test__raise_error_when_inversed_moment_name_is_not_found_in_input_keys():
    setting = IsoGCNSetting(
        nodes=[10, 20],
        isoam_names=["dummy"],
        self_network={"use_network": True},
        neumann_setting={
            "use_neumann": True,
            "factor": 0.2,
            "inversed_moment_source": "input_data",
            "inversed_moment_name": "moment_input",
            "neumann_input_name": "neumann_input",
        },
    )

    self_module = mock.MagicMock()
    self_module.get_input_keys.return_value = [
        "x",
        "neumann_input",
        "neumann_invalid_moment",
    ]

    with pytest.raises(
        ValueError, match="inversed_moment_name is not found in input_keys."
    ):
        setting.confirm(self_module)


# endregion


# region E2E tests

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/isogcn_setting"


@pytest.mark.parametrize(
    "yaml_file", ["check_isogcn_nodes.yml", "check_isogcn_neumann.yml"]
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
    "yaml_file, msg",
    [
        ("error_missing_neumann.yml", "Two inputs are necessary"),
        ("error_unnecessary_neumann.yml", "Only one input is allowed"),
        (
            "error_not_matched_neumann.yml",
            "neumann_input_name is not found in input_keys",
        ),
    ],
)
def test__raise_error_for_invalid_setting(yaml_file: str, msg: str):
    with open(_TEST_DATA_DIR / yaml_file) as fr:
        content = yaml.load(fr, Loader=yaml.SafeLoader)

    with pytest.raises(ValueError) as ex:
        setting = PhlowerModelSetting(**content["model"])
        setting.network.resolve(is_first=True)
        assert msg in str(ex)


# endregion
