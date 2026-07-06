import pathlib

import pytest
import yaml

from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import ResidualSetting


@pytest.mark.parametrize(
    "nodes, symbols_from_input, symbols_from_field, equation",
    [
        ([-1, 1], ["x", "y", "u"], ["c"], "Diff(u, x) + Diff(u, y) - c"),
        (
            [10, 1],
            ["x", "y", "z", "u"],
            ["c", "d"],
            "Diff(u, x) + Diff(u, y) + Diff(u, z) - c - d",
        ),
    ],
)
def test__can_accept_valid_n_nodes(
    nodes: list[int],
    symbols_from_input: list[str],
    symbols_from_field: list[str],
    equation: str,
):
    setting = ResidualSetting(
        nodes=nodes,
        symbols_from_input=symbols_from_input,
        symbols_from_field=symbols_from_field,
        equation=equation,
    )
    assert not setting.need_reference
    assert setting.get_n_nodes() == nodes


@pytest.mark.parametrize(
    "nodes, expected_msg",
    [
        ([10, 20, 30], "length of nodes must be 2"),
        ([10, -1], "nodes in Residual is inconsistent"),
    ],
)
def test__raise_error_when_invalid_n_nodes(nodes: list[int], expected_msg: str):
    with pytest.raises(ValueError, match=expected_msg):
        _ = ResidualSetting(
            nodes=nodes,
            symbols_from_input=["x", "y", "u"],
            symbols_from_field=["c"],
            equation="Diff(u, x) + Diff(u, y) - c",
        )


@pytest.mark.parametrize(
    "symbols_from_input, symbols_from_field, equation",
    [
        (
            ["x", "y"],  # missing u
            ["c"],
            "Diff(u, x) + Diff(u, y) - c",
        ),
        (
            ["x", "y", "u"],
            [],  # missing c
            "Diff(u, x) + Diff(u, y) - c + Diff(c, x)",
        ),
    ],
)
def test__raise_error_when_failed_to_parse_equation(
    symbols_from_input: list[str],
    symbols_from_field: list[str],
    equation: str,
):
    with pytest.raises(ValueError, match="Failed to parse equation"):
        _ = ResidualSetting(
            nodes=[-1, 1],
            symbols_from_input=symbols_from_input,
            symbols_from_field=symbols_from_field,
            equation=equation,
        )


@pytest.mark.parametrize(
    "symbols_from_input, symbols_from_field",
    [
        (
            ["x", "y", "u"],
            ["c", "u"],  # u is duplicated
        ),
        (
            ["x", "y", "c", "u"],  # c is duplicated
            ["c"],
        ),
    ],
)
def test__raise_error_when_symbols_are_not_unique(
    symbols_from_input: list[str],
    symbols_from_field: list[str],
):
    with pytest.raises(ValueError, match="must be unique"):
        _ = ResidualSetting(
            nodes=[-1, 1],
            symbols_from_input=symbols_from_input,
            symbols_from_field=symbols_from_field,
            equation="Diff(u, x) + Diff(u, y) - c",
        )


# region Test using yaml

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/residual_settings"


@pytest.mark.parametrize(
    "yaml_file",
    [
        "insufficient_input_keys.yml",
    ],
)
def test__raise_error_when_input_keys_are_not_enough(
    yaml_file: str,
):
    yaml_path = _TEST_DATA_DIR / yaml_file
    with open(yaml_path) as f:
        setting_dict = yaml.safe_load(f)
    setting = PhlowerModelSetting.model_validate(setting_dict["model"])

    with pytest.raises(ValueError, match="are not subset of actual inputs"):
        setting.resolve()


@pytest.mark.parametrize("yaml_file", ["check_nodes.yml"])
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
