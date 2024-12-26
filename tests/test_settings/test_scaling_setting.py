import pathlib
from unittest import mock

import pydantic
import pytest
from phlower.io import PhlowerDirectory
from phlower.settings import PhlowerScalingSetting
from phlower.settings._scaling_setting import (
    SameAsInputParameters,
    ScalerInputParameters,
    ScalerResolvedParameter,
)
from pipe import where

# region test for ScalerInputParameters


def test__always_join_fitting():
    with pytest.raises(pydantic.ValidationError):
        _ = ScalerInputParameters(method="identity", join_fitting=False)


def test__is_parent():
    scaler = ScalerInputParameters(method="identity")
    assert scaler.is_parent_scaler


@pytest.mark.parametrize(
    "method, variable_name, desired",
    [
        ("identity", "val_a", "SCALER_val_a"),
        ("user_defined", "val_b", "SCALER_val_b"),
    ],
)
def test_scaler_name(method: str, variable_name: str, desired: str):
    scaler = ScalerInputParameters(method=method)
    assert desired == scaler.get_scaler_name(variable_name)


# endregion

# region tests for SameAsInputParameters


def test__is_not_parent():
    scaler = SameAsInputParameters(same_as="val_a")
    assert not scaler.is_parent_scaler


@pytest.mark.parametrize("join_fitting", [True, False, None])
def test__join_fitting(join_fitting: bool):
    if join_fitting is None:
        scaler = SameAsInputParameters(same_as="val_a")
        assert not scaler.join_fitting
        return

    scaler = SameAsInputParameters(same_as="val_a", join_fitting=join_fitting)
    assert scaler.join_fitting == join_fitting


@pytest.mark.parametrize(
    "same_as, desired", [("val_a", "SCALER_val_a"), ("val_b", "SCALER_val_b")]
)
def test_sameas_scaler_name(same_as: str, desired: str):
    scaler = SameAsInputParameters(same_as=same_as)

    # NOTE: scaler name is decided by parent variable name
    assert scaler.get_scaler_name("aaa") == desired


# endregion

# region test for PhlowerScalingSetting


@pytest.mark.parametrize(
    "scalers",
    [
        (
            {
                "nodal": {"method": "std_scale"},
                "nodal_missing": {"same_as": "nodal_missing_parent"},
            }
        ),
        ({"nodal_1": {"same_as": "nodal_a"}}),
    ],
)
def test__missing_parent_scaler(scalers: dict):
    with pytest.raises(ValueError):
        _ = PhlowerScalingSetting(variable_name_to_scalers=scalers)


def test__read_yml():
    sample_file = (
        pathlib.Path(__file__).parent / "data/scaling/sample_setting.yml"
    )
    _ = PhlowerScalingSetting.read_yaml(sample_file)


@pytest.mark.parametrize(
    "scalers, desired",
    [
        (
            {
                "nodal": {"method": "std_scale"},
                "nodal_child": {"same_as": "nodal"},
            },
            ["nodal", "nodal_child"],
        ),
        (
            {
                "nodal": {"method": "std_scale"},
                "nodal2": {"method": "identity"},
            },
            ["nodal", "nodal2"],
        ),
        ({}, []),
    ],
)
def test__get_variable_names(
    scalers: dict[str, dict[str, str]], desired: list[str]
):
    scaler = PhlowerScalingSetting(variable_name_to_scalers=scalers)

    actual = scaler.get_variable_names()
    assert sorted(actual) == sorted(desired)


@pytest.mark.parametrize(
    "scalers, existed, not_existed",
    [
        (
            {
                "nodal": {"method": "std_scale"},
                "nodal_child": {"same_as": "nodal"},
            },
            ["nodal", "nodal_child"],
            ["nodal_v"],
        ),
        (
            {
                "nodal": {"method": "std_scale"},
                "nodal2": {"method": "identity"},
            },
            ["nodal", "nodal2"],
            ["nodal_child"],
        ),
    ],
)
def test__is_scaler_exist(
    scalers: dict[str, dict[str, str]],
    existed: list[str],
    not_existed: list[str],
):
    scaler = PhlowerScalingSetting(variable_name_to_scalers=scalers)

    for name in existed:
        assert scaler.is_scaler_exist(name)
    for name in not_existed:
        assert not scaler.is_scaler_exist(name)


@pytest.mark.parametrize(
    "scalers, desired",
    [
        (
            {
                "nodal": {"method": "std_scale"},
                "nodal_child": {"same_as": "nodal"},
            },
            [("nodal", "SCALER_nodal"), ("nodal_child", "SCALER_nodal")],
        ),
        (
            {
                "nodal": {"method": "std_scale"},
                "nodal2": {"method": "identity"},
            },
            [("nodal", "SCALER_nodal"), ("nodal2", "SCALER_nodal2")],
        ),
    ],
)
def test__get_scaler_name(
    scalers: dict[str, dict[str, str]], desired: list[tuple[str]]
):
    scaler = PhlowerScalingSetting(variable_name_to_scalers=scalers)

    for key, ans in desired:
        assert ans == scaler.get_scaler_name(key)


# endregion

# region test for ScalerResolvedParameter


@pytest.fixture
def create_sample_setting() -> PhlowerScalingSetting:
    scalers = {
        "nodal": {"method": "std_scale", "parameters": {"std_": 0.001}},
        "nodal_child": {"same_as": "nodal"},
        "nodal_child2": {"same_as": "nodal", "join_fitting": True},
        "value_x": {"method": "identity", "component_wise": False},
        "value_y": {"method": "standardize", "component_wise": True},
        "value_z": {
            "method": "user_defined",
            "parameters": {"user_std_": 10.0},
        },
    }
    return PhlowerScalingSetting(variable_name_to_scalers=scalers)


def test__n_resolved_settings(create_sample_setting: PhlowerScalingSetting):
    scaler: PhlowerScalingSetting = create_sample_setting

    resolved = scaler.resolve_scalers()
    assert len(resolved) == 4


@pytest.mark.parametrize(
    "scaler_name, desired",
    [
        ("SCALER_nodal", ["nodal", "nodal_child", "nodal_child2"]),
        ("SCALER_value_x", ["value_x"]),
        ("SCALER_value_z", ["value_z"]),
    ],
)
def test__transform_items(
    scaler_name: str,
    desired: list[str],
    create_sample_setting: PhlowerScalingSetting,
):
    scaler = create_sample_setting
    resolved = scaler.resolve_scalers()

    target = list(resolved | where(lambda x: x.scaler_name == scaler_name))[0]

    assert target.transform_members == desired


@pytest.mark.parametrize(
    "scaler_name, desired",
    [
        ("SCALER_nodal", ["nodal", "nodal_child2"]),
        ("SCALER_value_x", ["value_x"]),
        ("SCALER_value_z", ["value_z"]),
    ],
)
def test__fitting_items(
    scaler_name: str, desired: list[str], create_sample_setting: str
):
    scaler: PhlowerScalingSetting = create_sample_setting
    resolved = scaler.resolve_scalers()

    target = list(resolved | where(lambda x: x.scaler_name == scaler_name))[0]

    assert target.fitting_members == desired


@pytest.mark.parametrize(
    "scaler_name, desired",
    [
        ("SCALER_nodal", False),
        ("SCALER_value_x", False),
        ("SCALER_value_y", True),
    ],
)
def test__component_wise(
    scaler_name: str, desired: str, create_sample_setting: PhlowerScalingSetting
):
    scaler: PhlowerScalingSetting = create_sample_setting
    resolved = scaler.resolve_scalers()

    target = list(resolved | where(lambda x: x.scaler_name == scaler_name))[0]

    assert target.component_wise == desired


@pytest.mark.parametrize(
    "scaler_name, desired",
    [
        ("SCALER_nodal", {"std_": 0.001}),
        ("SCALER_value_z", {"user_std_": 10.0}),
    ],
)
def test__parameters(
    scaler_name: str,
    desired: dict[str, float],
    create_sample_setting: PhlowerScalingSetting,
):
    scaler: PhlowerScalingSetting = create_sample_setting
    resolved = scaler.resolve_scalers()

    target = list(resolved | where(lambda x: x.scaler_name == scaler_name))[0]

    assert target.parameters == desired


@pytest.mark.parametrize(
    "scaler_name, allow_missing",
    [("SCALER_nodal", True), ("SCALER_value_z", False)],
)
def test__collect_fitting_files(
    scaler_name: str,
    allow_missing: bool,
    create_sample_setting: PhlowerScalingSetting,
):
    scaler: PhlowerScalingSetting = create_sample_setting
    resolved = scaler.resolve_scalers()

    target: ScalerResolvedParameter = list(
        resolved | where(lambda x: x.scaler_name == scaler_name)
    )[0]

    with mock.patch.object(PhlowerDirectory, "find_variable_file") as mocked:
        _ = target.collect_fitting_files(
            directory=pathlib.Path("dummy"), allow_missing=allow_missing
        )

        assert len(mocked.mock_calls) == len(target.fitting_members)
        for _, args, kwargs in mocked.mock_calls:
            assert len(args) == 1
            assert kwargs.get("allow_missing") == allow_missing

        args = [x[0] for _, x, _ in mocked.mock_calls]
        assert args == target.fitting_members


@pytest.mark.parametrize(
    "scaler_name, allow_missing",
    [("SCALER_nodal", True), ("SCALER_value_z", False)],
)
def test__collect_transform_files(
    scaler_name: str,
    allow_missing: bool,
    create_sample_setting: PhlowerScalingSetting,
):
    scaler: PhlowerScalingSetting = create_sample_setting
    resolved = scaler.resolve_scalers()

    target: ScalerResolvedParameter = list(
        resolved | where(lambda x: x.scaler_name == scaler_name)
    )[0]

    with mock.patch.object(PhlowerDirectory, "find_variable_file") as mocked:
        _ = target.collect_transform_files(
            directory=pathlib.Path("dummy"), allow_missing=allow_missing
        )

        assert len(mocked.mock_calls) == len(target.transform_members)
        for _, args, kwargs in mocked.mock_calls:
            assert len(args) == 1
            assert kwargs.get("allow_missing") == allow_missing

        args = [x[0] for _, x, _ in mocked.mock_calls]
        assert args == target.transform_members


# endregion


@pytest.mark.parametrize(
    "scalers",
    [
        (
            {
                "nodal_hop": {
                    "method": "isoam_scale",
                    "parameters": {"other_components": []},
                },
            }
        ),
        (
            {
                "nodal_hopx": {
                    "method": "isoam_scale",
                    "parameters": {
                        "other_components": ["nodal_hopy", "nodal_hopz"]
                    },
                },
                "nodal_hopy": {"same_as": "nodal_hopx", "join_fitting": False},
                "nodal_hopz": {"same_as": "nodal_hopx", "join_fitting": False},
            }
        ),
    ],
)
def test__validate_isoam(scalers: dict):
    setting = PhlowerScalingSetting(variable_name_to_scalers=scalers)
    with pytest.raises(ValueError):
        _ = setting.resolve_scalers()[0]
