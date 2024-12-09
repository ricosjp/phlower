import pydantic
import pytest
import yaml
from phlower.settings import GroupModuleSetting


@pytest.mark.parametrize("solver_type", ["newton", "aaa", "bbb"])
def test__not_implemeted_iteration_solver_name(solver_type: str):
    with pytest.raises(NotImplementedError) as ex:
        _ = GroupModuleSetting(
            name="Sample",
            inputs=[],
            outputs=[],
            modules=[],
            destinations=[],
            solver_type=solver_type,
        )
    assert f"solver_type: {solver_type} is not implemented." in str(ex)


@pytest.mark.parametrize(
    "yaml_file",
    [
        "tests/test_settings/data/groups/target_missing_solver_1.yml",
        "tests/test_settings/data/groups/target_missing_solver_2.yml",
        "tests/test_settings/data/groups/target_missing_solver_3.yml",
    ],
)
def test__check_solver_target_name_is_missing(yaml_file: str):
    with open(yaml_file) as fr:
        data = yaml.load(fr, Loader=yaml.SafeLoader)

    with pytest.raises(ValueError) as ex:
        _ = GroupModuleSetting(**data["network"])

        assert "is missing" in str(ex)


@pytest.mark.parametrize(
    "yaml_file",
    [
        "tests/test_settings/data/groups/extra_keywords.yml",
    ],
)
def test__raise_error_when_extra_keyword_exists(yaml_file: str):
    with open(yaml_file) as fr:
        data = yaml.load(fr, Loader=yaml.SafeLoader)

    with pytest.raises(pydantic.ValidationError) as ex:
        _ = GroupModuleSetting(**data["network"])

        assert "Extra inputs are not permitted" in str(ex)
