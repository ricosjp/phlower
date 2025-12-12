import pathlib

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import given

from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import RearrangeSetting

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/rearrange_settings"


@pytest.mark.parametrize(
    "pattern, axes_lengths",
    [
        ("time vertex dim (f1 f2) -> (time f1) vertex dim f2", {"f2": 20}),
        ("time vertex dim f1 -> f1 vertex dim time", {}),
    ],
)
@given(
    n_last_feature=st.integers(min_value=1),
    save_as_time_series=st.booleans(),
    save_as_voxel=st.booleans(),
)
def test__can_accept_valid_n_nodes(
    pattern: str,
    axes_lengths: dict[str, int],
    n_last_feature: int,
    save_as_time_series: bool,
    save_as_voxel: bool,
):
    setting = RearrangeSetting(
        pattern=pattern,
        axes_lengths=axes_lengths,
        output_feature_dim=n_last_feature,
        save_as_time_series=save_as_time_series,
        save_as_voxel=save_as_voxel,
    )

    assert setting.pattern == pattern
    assert setting.axes_lengths == axes_lengths
    assert setting.output_feature_dim == setting.output_feature_dim
    assert setting.save_as_time_series == save_as_time_series
    assert setting.save_as_voxel == save_as_voxel


@pytest.mark.parametrize(
    "nodes, desired_msg",
    [
        ([10, 20, 30], "length of nodes must be 2."),
        ([-1, -1], "value -1 is not allowed except the head node."),
    ],
)
def test__raise_error_when_invalid_n_nodes(nodes: list[int], desired_msg: str):
    with pytest.raises(ValueError) as ex:
        _ = RearrangeSetting(nodes=nodes)
    assert desired_msg in str(ex.value)


@pytest.mark.parametrize("input_dims", [([30]), ([40]), ([100])])
def test__gather_input_dims(input_dims: list[int]):
    setting = RearrangeSetting(
        pattern="time vertex dim f1 -> f1 vertex dim time",
        output_feature_dim=10,
        save_as_time_series=False,
        save_as_voxel=False,
    )

    assert setting.gather_input_dims(*input_dims) == input_dims[0]


@pytest.mark.parametrize("input_dims", [([]), ([40, 400]), ([10, 0, 1])])
def test__raise_error_invalid_input_dims(input_dims: list[int]):
    setting = RearrangeSetting(
        pattern="time vertex dim f1 -> f1 vertex dim time",
        output_feature_dim=10,
    )

    with pytest.raises(ValueError) as ex:
        _ = setting.gather_input_dims(*input_dims)
    assert "only one input is allowed" in str(ex.value)


@pytest.mark.parametrize("n_last_feature", [10, 20, 33])
@given(input_ndim=st.integers(min_value=1))
def test__nodes_after_overwriting(n_last_feature: int, input_ndim: int):
    setting = RearrangeSetting(
        pattern="time vertex dim f1 -> f1 vertex dim time",
        output_feature_dim=n_last_feature,
    )

    desired = [input_ndim, n_last_feature]

    before_nodes = setting.get_n_nodes()
    assert before_nodes != desired

    setting.overwrite_nodes(desired)
    assert setting.get_n_nodes() == desired


@pytest.mark.parametrize(
    "n_last_feature, update_nodes, desired_msg",
    [
        (10, [-1, 10], "Resolved nodes must be positive"),
        (10, [10, 100], "the value of n_last_feature and"),
        (20, [10, 100, 30], "Invalid length of nodes"),
        (20, [5, 100, 20], "Invalid length of nodes"),
    ],
)
def test__cannot_update_by_invalid_nodes(
    n_last_feature: int, update_nodes: list[int], desired_msg: str
):
    setting = RearrangeSetting(
        pattern="time vertex dim f1 -> f1 vertex dim time",
        output_feature_dim=n_last_feature,
    )

    with pytest.raises(ValueError) as ex:
        setting.overwrite_nodes(update_nodes)

    assert desired_msg in str(ex.value)


def test__reference_is_not_necessary():
    setting = RearrangeSetting(
        pattern="time vertex dim f1 -> f1 vertex dim time",
        output_feature_dim=10,
    )

    assert not setting.need_reference


# region E2E tests only for ReducerSettings

_TEST_DATA_DIR = pathlib.Path(__file__).parent / "data/rearrange_setting"


@pytest.mark.parametrize("yaml_file", ["check_rearrange_nodes.yml"])
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
