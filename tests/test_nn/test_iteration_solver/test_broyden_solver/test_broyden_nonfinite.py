import pytest
import torch
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn._interface_iteration_solver import IOptimizeProblem
from phlower.nn._iteration_solvers._broyden_solver import BroydenSolver


class ConstantProblem(IOptimizeProblem):
    def __init__(self, residuals: dict[str, torch.Tensor]) -> None:
        self.residuals = phlower_tensor_collection(
            {k: phlower_tensor(v) for k, v in residuals.items()}
        )
        self.calls = 0

    def step_forward(
        self, h: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:
        return h

    def gradient(
        self,
        h: IPhlowerTensorCollections,
        update_keys: list[str],
        operator_keys: list[str] | None = None,
    ) -> IPhlowerTensorCollections:
        self.calls += 1
        return self.residuals


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("stage", ["initial values", "residual"])
def test_rejects_one_nonfinite_key(bad: float, stage: str) -> None:
    solver = BroydenSolver(3, 1e-5, 100, update_keys=["x", "y"])
    initial = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.tensor([0.0])),
            "y": phlower_tensor(
                torch.tensor([bad if stage == "initial values" else 0.0])
            ),
        }
    )
    problem = ConstantProblem(
        {
            "x": torch.tensor([1.0]),
            "y": torch.tensor([bad if stage == "residual" else 1.0]),
        }
    )
    with pytest.raises(FloatingPointError, match=f"{stage}.*'y'.*iteration"):
        solver.run(initial, problem)
    assert not solver.get_converged()
    assert problem.calls == (0 if stage == "initial values" else 1)


def test_rejects_nonfinite_norm_of_finite_residual() -> None:
    solver = BroydenSolver(3, 1e-5, 100, update_keys=["x"])
    residual = torch.tensor([3e38, 3e38], dtype=torch.float32)
    assert torch.isfinite(residual).all()
    assert not torch.isfinite(torch.linalg.norm(residual))
    problem = ConstantProblem({"x": residual})
    initial = phlower_tensor_collection({"x": phlower_tensor(torch.zeros(2))})
    with pytest.raises(FloatingPointError, match="residual norm.*'x'"):
        solver.run(initial, problem)
    assert problem.calls == 1


@pytest.mark.parametrize("max_iterations", [1, 3])
def test_rejects_overflow_in_adopted_update(max_iterations: int) -> None:
    solver = BroydenSolver(max_iterations, 1e-5, 3.3e38, update_keys=["x"])
    problem = ConstantProblem({"x": torch.tensor([1e38], dtype=torch.float32)})
    initial = phlower_tensor_collection(
        {"x": phlower_tensor(torch.tensor([-3e38], dtype=torch.float32))}
    )
    with pytest.raises(
        FloatingPointError, match="updated values.*'x'.*iteration 1"
    ):
        solver.run(initial, problem)
    assert not solver.get_converged()
    assert problem.calls == 1


def test_returns_pre_update_value_when_diverged_update_is_discarded() -> None:
    solver = BroydenSolver(
        1,
        1e-5,
        100,
        update_keys=["x"],
        exit_before_update_when_diverged=True,
    )
    problem = ConstantProblem({"x": torch.tensor([1e38], dtype=torch.float32)})
    initial = phlower_tensor_collection(
        {"x": phlower_tensor(torch.tensor([-3e38], dtype=torch.float32))}
    )
    result = solver.run(initial, problem)
    torch.testing.assert_close(
        result["x"].to_tensor(), initial["x"].to_tensor()
    )
    assert solver._iterated_state.is_diverged


def test_preserves_finite_final_update_and_gradient() -> None:
    theta = torch.tensor([1e-6], dtype=torch.float64, requires_grad=True)
    problem = ConstantProblem({"x": -theta})
    solver = BroydenSolver(3, 1e-5, 100, update_keys=["x"])
    initial = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.zeros(1, dtype=torch.float64)),
            "unused_boundary": phlower_tensor(torch.tensor([float("nan")])),
        }
    )
    result = solver.run(initial, problem)["x"].to_tensor()
    assert solver.get_converged()
    assert problem.calls == 1
    torch.testing.assert_close(result, theta)
    (gradient,) = torch.autograd.grad(result.sum(), theta)
    torch.testing.assert_close(gradient, torch.ones_like(theta))
