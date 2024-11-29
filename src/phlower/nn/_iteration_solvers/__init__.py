from phlower.nn._interface_iteration_solver import IFIterationSolver
from phlower.nn._iteration_solvers._barzilai_borwein_solver import (
    BarzilaiBorweinSolver,
)
from phlower.nn._iteration_solvers._empty_solver import EmptySolver
from phlower.nn._iteration_solvers._simple_iteration_solver import (
    SimpleIterationSolver,
)
from phlower.utils.enums import PhlowerIterationSolverType

_name2solver: dict[str, IFIterationSolver] = {
    PhlowerIterationSolverType.none.name: EmptySolver,
    PhlowerIterationSolverType.simple.name: SimpleIterationSolver,
    PhlowerIterationSolverType.barzilai_borwein.name: BarzilaiBorweinSolver,
}


def get_iteration_solver(name: str) -> IFIterationSolver:
    if name in _name2solver:
        raise NotImplementedError(
            f"Iteration solver named {name} is not implemented."
        )
    return _name2solver[name]
