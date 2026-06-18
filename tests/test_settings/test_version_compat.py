import pathlib

from phlower.settings import PhlowerSetting
from phlower.settings._iteration_solver_setting import (
    BarzilaiBoweinSolverSetting,
)

DATA_DIR = pathlib.Path("tests/test_settings/data/backward_compatibility")

# region: skip_last_update for backward compatibility


def test__backward_compatibility_of_skip_last_update():
    setting = PhlowerSetting.read_yaml(DATA_DIR / "bb_setting.yml")

    nn_model = setting.model.network
    assert nn_model.solver_type == "bb"
    assert isinstance(nn_model.solver_parameters, BarzilaiBoweinSolverSetting)
    assert nn_model.solver_parameters.skip_last_update is True


# endregion
