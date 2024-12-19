import pydantic
import pytest
from phlower.settings._nonlinear_solver_setting import (
    BarzilaiBoweinSolverSetting,
    EmptySolverSetting,
    SimpleSolverSetting,
)


def test__empty_setting():
    setting = EmptySolverSetting()
    assert len(setting.get_target_keys()) == 0


@pytest.mark.parametrize(
    "convergence_threshold, max_iterations",
    [(-0.001, 100), (100, -123), (-100, -20)],
)
def test__raise_error_for_invalid_args_in_simple_setting(
    convergence_threshold: float, max_iterations: int
):
    with pytest.raises(pydantic.ValidationError) as ex:
        _ = SimpleSolverSetting(
            target_keys=["sample"],
            convergence_threshold=convergence_threshold,
            max_iterations=max_iterations,
        )

        assert "must be positive value" in str(ex)


def test__raise_error_when_empty_targets_in_simple_setting():
    with pytest.raises(pydantic.ValidationError) as ex:
        _ = SimpleSolverSetting(
            target_keys=[],
        )

        assert "Set at least one target item" in str(ex)


# region test for Barazilai Borwein setting


@pytest.mark.parametrize(
    "convergence_threshold, divergence_threshold, max_iterations",
    [
        (10, -0.001, 100),
        (2, 100, -123),
        (3, -100, -20),
        (-0.210, 0.001, 100),
        (-3, 100, -20),
    ],
)
def test__raise_error_for_invalid_args_in_bb_setting(
    convergence_threshold: float,
    divergence_threshold: float,
    max_iterations: int,
):
    with pytest.raises(pydantic.ValidationError) as ex:
        _ = BarzilaiBoweinSolverSetting(
            target_keys=["sample"],
            convergence_threshold=convergence_threshold,
            divergence_threshold=divergence_threshold,
            max_iterations=max_iterations,
        )

        assert "must be positive value" in str(ex)


@pytest.mark.parametrize(
    "convergence_threshold, divergence_threshold",
    [(1000, 0.00001), (10000, 10000), (0.333, 0.222)],
)
def test__raise_error_when_convergence_is_greater_than_divergenve(
    convergence_threshold: float, divergence_threshold: float
):
    with pytest.raises(pydantic.ValidationError) as ex:
        _ = BarzilaiBoweinSolverSetting(
            target_keys=["sample"],
            convergence_threshold=convergence_threshold,
            divergence_threshold=divergence_threshold,
        )

        assert "convergence threshold must be less than d" in str(ex)


def test__raise_error_when_empty_targets_in_bb_setting():
    with pytest.raises(pydantic.ValidationError) as ex:
        _ = BarzilaiBoweinSolverSetting(
            target_keys=[],
        )

        assert "Set at least one target item" in str(ex)


@pytest.mark.parametrize("bb_type", ["long_short", "sample", "aaaa"])
def test__raise_error_when_invalid_bb_type(bb_type: str):
    with pytest.raises(pydantic.ValidationError):
        _ = BarzilaiBoweinSolverSetting(target_keys=["aaa"], bb_type=bb_type)


# endregion
