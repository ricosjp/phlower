import pydantic
import pytest

from phlower.settings._module_settings import EdgeGatherSetting


@pytest.mark.parametrize("nodes", [(None), ([10, 20])])
def test__can_accept_valid_n_nodes(nodes: list[int] | None):
    _ = EdgeGatherSetting(support_name="support", nodes=nodes)


@pytest.mark.parametrize("nodes", [([5]), ([10, 10, 10])])
def test__raise_error_when_invalid_n_nodes(nodes: list[int]):
    with pytest.raises(pydantic.ValidationError) as ex:
        _ = EdgeGatherSetting(support_name="support", nodes=nodes)

    assert "Size of nodes must be 2" in str(ex.value)


@pytest.mark.parametrize("input_dims, desired", [([10], 10), ([3], 3)])
def test__gather_input_dims(input_dims: list[int], desired: int):
    setting = EdgeGatherSetting(support_name="support")

    assert setting.gather_input_dims(*input_dims) == desired


@pytest.mark.parametrize("input_dims", [([]), ([10, 3])])
def test__raise_error_when_invalid_input_dims(input_dims: list[int]):
    setting = EdgeGatherSetting(support_name="support")

    with pytest.raises(ValueError) as ex:
        setting.gather_input_dims(*input_dims)

    assert "num of input should be 1" in str(ex.value)


@pytest.mark.parametrize(
    "input_dims, desired", [([3], [3, 6]), ([10], [10, 20])]
)
def test__get_default_nodes(input_dims: list[int], desired: list[int]):
    setting = EdgeGatherSetting(support_name="support")

    assert setting.get_default_nodes(*input_dims) == desired


def test__nodes_is_update_after_overwrite_nodes():
    setting = EdgeGatherSetting(support_name="support")

    assert setting.get_n_nodes() is None

    setting.overwrite_nodes([3, 6])
    assert setting.get_n_nodes() == [3, 6]


def test__reference_is_not_necessary():
    setting = EdgeGatherSetting(support_name="support")

    assert not setting.need_reference
