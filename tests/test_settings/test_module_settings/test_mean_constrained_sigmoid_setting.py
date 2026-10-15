import pathlib

import hypothesis.strategies as st
import pytest
import yaml
from hypothesis import given

from phlower.settings import PhlowerModelSetting
from phlower.settings._module_settings import MeanConstrainedSigmoidSetting


@pytest.mark.parametrize("reference_variable_name", ["a", "b"])
@pytest.mark.parametrize("weight_field_name", [None, "w"])
@pytest.mark.parametrize("slope", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("learnable_slope", [True, False])
@given(
    output_feature_dim=st.integers(min_value=1),
)
def test__can_set_conditions(
    output_feature_dim: int,
    reference_variable_name: str,
    weight_field_name: str | None,
    slope: float,
    learnable_slope: bool,
):
    setting = MeanConstrainedSigmoidSetting(
        output_feature_dim=output_feature_dim,
        reference_variable_name=reference_variable_name,
        weight_field_name=weight_field_name,
        slope=slope,
        learnable_slope=learnable_slope,
    )

    assert setting.output_feature_dim == output_feature_dim
    assert setting.reference_variable_name == reference_variable_name
    assert setting.weight_field_name == weight_field_name
    assert setting.slope == slope
    assert setting.learnable_slope == learnable_slope


@pytest.mark.parametrize(
    "nodes, output_feature_dim, desired_msg",
    [
        ([10, 20, 30], 30, "length of nodes must be 2."),
        ([-1, -1], -1, "value -1 is not allowed except the head node."),
    ],
)
def test__raise_error_when_invalid_n_nodes(
    nodes: list[int], output_feature_dim: int, desired_msg: str
):
    with pytest.raises(ValueError) as ex:
        _ = MeanConstrainedSigmoidSetting(
            nodes=nodes,
            output_feature_dim=output_feature_dim,
            reference_variable_name="a",
        )
    assert desired_msg in str(ex.value)


@pytest.mark.parametrize("input_dims", [([30]), ([40]), ([100])])
def test__gather_input_dims(input_dims: list[int]):
    setting = MeanConstrainedSigmoidSetting(
        output_feature_dim=10,
        reference_variable_name="a",
    )

    assert setting.gather_input_dims(*input_dims) == input_dims[0]


@pytest.mark.parametrize("output_feature_dim", [10, 20, 33])
@given(input_ndim=st.integers(min_value=1))
def test__nodes_after_overwriting(output_feature_dim: int, input_ndim: int):
    setting = MeanConstrainedSigmoidSetting(
        output_feature_dim=output_feature_dim,
        reference_variable_name="a",
    )

    desired = [input_ndim, output_feature_dim]

    before_nodes = setting.get_n_nodes()
    assert before_nodes != desired

    setting.overwrite_nodes(desired)
    assert setting.get_n_nodes() == desired


@pytest.mark.parametrize(
    "output_feature_dim, update_nodes, desired_msg",
    [
        (10, [-1, 10], "Resolved nodes must be positive"),
        (10, [10, 100], "the value of output_feature_dim and"),
        (20, [10, 100, 30], "Invalid length of nodes"),
        (20, [5, 100, 20], "Invalid length of nodes"),
    ],
)
def test__cannot_update_by_invalid_nodes(
    output_feature_dim: int, update_nodes: list[int], desired_msg: str
):
    setting = MeanConstrainedSigmoidSetting(
        output_feature_dim=output_feature_dim,
        reference_variable_name="a",
    )

    with pytest.raises(ValueError) as ex:
        setting.overwrite_nodes(update_nodes)

    assert desired_msg in str(ex.value)


def test__reference_is_not_necessary():
    setting = MeanConstrainedSigmoidSetting(
        output_feature_dim=10,
        reference_variable_name="a",
    )

    assert not setting.need_reference


# region E2E tests only for ReducerSettings

_TEST_DATA_DIR = (
    pathlib.Path(__file__).parent / "data/mean_constrained_sigmoid_setting"
)


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
