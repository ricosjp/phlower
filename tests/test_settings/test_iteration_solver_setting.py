import pydantic
import pytest
from phlower.settings._iteration_solver_setting import (
    BarzilaiBoweinSolverSetting,
    ConjugateGradientSolverSetting,
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


def test__target_keys_is_deprecated_and_correctly_passed():
    setting = BarzilaiBoweinSolverSetting(
        target_keys=["a", "b", "c"],
        convergence_threshold=0.1,
        max_iterations=10,
        divergence_threshold=50,
    )

    assert setting.update_keys == ["a", "b", "c"]
    assert setting.operator_keys == ["a", "b", "c"]

    with pytest.deprecated_call():
        _ = setting.target_keys


def test__update_and_residual_keys():
    setting = BarzilaiBoweinSolverSetting(
        update_keys=["a", "b"],
        operator_keys=["c", "d", "e"],
        convergence_threshold=0.1,
        max_iterations=10,
        divergence_threshold=50,
    )

    assert setting.update_keys == ["a", "b"]
    assert setting.operator_keys == ["c", "d", "e"]


# endregion

# region test for conjugate gradient setting


def test__cg_empty_list_is_not_allowed():
    with pytest.raises(pydantic.ValidationError, match="update_keys"):
        ConjugateGradientSolverSetting(update_keys=[])


def test__cg_convergence_must_be_less_than_divergence():
    with pytest.raises(
        pydantic.ValidationError,
        match="convergence threshold must be",
    ):
        ConjugateGradientSolverSetting(
            update_keys=["a"],
            convergence_threshold=0.1,
            divergence_threshold=0.01,
        )


# endregion
