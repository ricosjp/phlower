import pathlib

import pydantic
import pytest
import yaml

from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import EdgeDifferenceSetting


@pytest.mark.parametrize("nodes", [(None), ([10, 10])])
def test__can_accept_valid_n_nodes(nodes: list[int] | None):
    _ = EdgeDifferenceSetting(support_name="support", nodes=nodes)


@pytest.mark.parametrize("nodes", [([5]), ([10, 10, 10])])
def test__raise_error_when_invalid_n_nodes(nodes: list[int]):
    with pytest.raises(pydantic.ValidationError) as ex:
        _ = EdgeDifferenceSetting(support_name="support", nodes=nodes)

    assert "Size of nodes must be 2" in str(ex.value)


@pytest.mark.parametrize("input_dims, desired", [([10], 10), ([3], 3)])
def test__gather_input_dims(input_dims: list[int], desired: int):
    setting = EdgeDifferenceSetting(support_name="support")

    assert setting.gather_input_dims(*input_dims) == desired


@pytest.mark.parametrize("input_dims", [([]), ([10, 3])])
def test__raise_error_when_invalid_input_dims(input_dims: list[int]):
    setting = EdgeDifferenceSetting(support_name="support")

    with pytest.raises(ValueError) as ex:
        setting.gather_input_dims(*input_dims)

    assert "num of input should be 1" in str(ex.value)


@pytest.mark.parametrize(
    "with_norm, input_dims, desired",
    [(False, [3], [3, 3]), (True, [3], [3, 4]), (True, [10], [10, 11])],
)
def test__get_default_nodes(
    with_norm: bool, input_dims: list[int], desired: list[int]
):
    setting = EdgeDifferenceSetting(support_name="support", with_norm=with_norm)

    assert setting.get_default_nodes(*input_dims) == desired


def test__nodes_is_update_after_overwrite_nodes():
    setting = EdgeDifferenceSetting(support_name="support")

    assert setting.get_n_nodes() is None

    setting.overwrite_nodes([3, 4])
    assert setting.get_n_nodes() == [3, 4]


def test__reference_is_not_necessary():
    setting = EdgeDifferenceSetting(support_name="support")

    assert not setting.need_reference


# region E2E tests

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/edge_setting"


@pytest.mark.parametrize("yaml_file", ["check_edge_nodes.yml"])
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
