from phlower.collections import IPhlowerTensorCollections
from phlower.nn._interface_iteration_solver import (
    IFIterationSolver,
    IOptimizeProblem,
)


class EmptySolver(IFIterationSolver):
    def __init__(self) -> None: ...

    def zero_residuals(self) -> None:
        return

    def get_converged(self) -> bool:
        return True

    def run(
        self,
        initial_values: IPhlowerTensorCollections,
        problem: IOptimizeProblem,
    ) -> IPhlowerTensorCollections:
        return problem.objective(initial_values)
