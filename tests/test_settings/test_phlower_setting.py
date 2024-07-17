import pydantic
import pytest

from phlower.settings import PhlowerModelSetting, PhlowerPredictorSetting


@pytest.mark.parametrize(
    "variable_dimensions",
    [({"vala": {"mass_": 1, "time": 4}}), ({"sample": {"kg": 2, "Am": -2}})],
)
def test__invalid_dimension_model_setting(variable_dimensions):
    with pytest.raises(TypeError):
        _ = PhlowerModelSetting(
            variable_dimensions=variable_dimensions,
            network={
                "name": "sample",
                "inputs": [],
                "outputs": [],
                "modules": [],
            },
        )


@pytest.mark.parametrize(
    "selection_mode", ["best", "latest", "train_best", "specified"]
)
def test__valid_selection_mode(selection_mode):
    _ = PhlowerPredictorSetting(selection_mode=selection_mode)


@pytest.mark.parametrize("selection_mode", ["other", "best_of_best"])
def test__invalid_selection_mode(selection_mode):
    with pytest.raises(pydantic.ValidationError):
        _ = PhlowerPredictorSetting(selection_mode=selection_mode)
