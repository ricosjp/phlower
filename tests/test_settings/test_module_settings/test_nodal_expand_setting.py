import pathlib

import pytest
import yaml

from phlower.nn import NodalExpand
from phlower.nn._core_modules import get_module
from phlower.settings import PhlowerModelSetting
from phlower.settings._module_setting import ModuleSetting
from phlower.settings._module_settings import NodalExpandSetting


@pytest.mark.parametrize(
    "input_dims", [{"source": 3, "mesh": 7}, {"mesh": 7, "source": 3}]
)
def test_source_feature_dimension_is_preserved(input_dims: dict[str, int]):
    setting = NodalExpandSetting(reference_key="mesh")
    assert setting.gather_input_dims(input_dims) == 3
    assert setting.get_default_nodes(input_dims) == [3, 3]
    assert not setting.need_reference
    setting.overwrite_nodes([3, 3])
    assert setting.get_n_nodes() == [3, 3]


@pytest.mark.parametrize(
    "input_dims", [{}, {"source": 3}, {"source": 3, "reference": 7, "extra": 2}]
)
def test_reject_invalid_input_count(input_dims: dict[str, int]):
    with pytest.raises(ValueError, match="Exactly two inputs"):
        NodalExpandSetting(reference_key="reference").gather_input_dims(
            input_dims
        )


@pytest.mark.parametrize("nodes", [[], [3], [3, 3, 3]])
def test_reject_invalid_nodes(nodes: list[int]):
    with pytest.raises(ValueError, match="Size of nodes must be 2"):
        NodalExpandSetting(reference_key="reference", nodes=nodes)


@pytest.mark.parametrize(
    "input_keys", [["source", "reference"], ["reference", "source"]]
)
def test_registered_module_and_feature_resolution(input_keys: list[str]):
    setting = ModuleSetting(
        name="expand",
        nn_type="NodalExpand",
        input_keys=input_keys,
        output_key="expanded",
        nn_parameters={"reference_key": "reference", "unbatch_key": "mesh"},
    )
    assert isinstance(setting.nn_parameters, NodalExpandSetting)
    # The named reference is excluded from the output feature dimension.
    assert setting._resolve_nodes({"reference": 7, "source": 3}) == [3, 3]
    assert get_module("NodalExpand") is NodalExpand
    model = get_module("NodalExpand").from_setting(setting.nn_parameters)
    assert model._reference_key == "reference"
    assert model._unbatch_key == "mesh"


def test_reference_key_is_required():
    with pytest.raises(ValueError, match="reference_key"):
        NodalExpandSetting()


def test_reject_missing_reference_in_feature_resolution():
    setting = NodalExpandSetting(reference_key="missing")
    with pytest.raises(ValueError, match="reference_key 'missing'"):
        setting.get_default_nodes({"source": 3, "reference": 7})


def test_confirm_rejects_missing_reference_key():
    setting = ModuleSetting(
        name="expand",
        nn_type="NodalExpand",
        input_keys=["source", "reference"],
        nn_parameters={"reference_key": "missing"},
    )
    with pytest.raises(ValueError, match="reference_key 'missing'"):
        setting.nn_parameters.confirm(setting)


# region E2E tests only for MLPSettings

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/nodal_expand_setting"


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
