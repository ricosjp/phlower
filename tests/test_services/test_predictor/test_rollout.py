import pathlib
import shutil
from typing import NamedTuple

import numpy as np
import pytest
import yaml

from phlower.services.predictor import PhlowerPredictor
from phlower.services.predictor._rollout import (
    _determine_n_steps,
    _resolve_rollout_variables,
)
from phlower.services.preprocessing import PhlowerScalingService
from phlower.services.trainer import PhlowerTrainer
from phlower.settings import (
    PhlowerModelSetting,
    PhlowerPredictorSetting,
    PhlowerScalingSetting,
    PhlowerSetting,
    RolloutSetting,
)

_TRAIN_YAML = pathlib.Path(__file__).parent / "data/train_rollout.yml"
_PREDICT_YAML = pathlib.Path(__file__).parent / "data/predict_rollout.yml"
_N_TIMES = 6
_U_AND_P = [
    {"state_variable": "u", "increment_variable": "du"},
    {"state_variable": "p", "increment_variable": "dp"},
]
_ONLY_U = _U_AND_P[:1]


def _load_yaml(path: pathlib.Path) -> dict:
    with open(path) as fr:
        return yaml.safe_load(fr)


def _model_setting() -> dict:
    return _load_yaml(_TRAIN_YAML)["model"]


def _prediction_setting(**rollout) -> dict:
    """Prediction setting of the yml, with the given items of rollout."""
    setting = _load_yaml(_PREDICT_YAML)["prediction"]
    setting["rollout"] |= rollout
    return setting


class _TrainedModel(NamedTuple):
    model_directory: pathlib.Path
    interim_directories: list[pathlib.Path]
    preprocessed_directories: list[pathlib.Path]
    scaling_yaml: pathlib.Path


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> _TrainedModel:
    base = tmp_path_factory.mktemp("rollout")
    interim_directories = []
    for i, n_nodes in enumerate([10, 20]):
        directory = base / f"interim/case_{i}"
        directory.mkdir(parents=True)
        u = np.random.randn(_N_TIMES, n_nodes, 2) * np.array([1.0, 3.0])
        p = np.random.randn(_N_TIMES, n_nodes, 1) * 5.0
        for name, value in [("u", u), ("p", p)]:
            np.save(directory / f"nodal_{name}.npy", value.astype(np.float32))
            np.save(
                directory / f"nodal_d{name}.npy",
                (value[1:] - value[:-1]).astype(np.float32),
            )
        interim_directories.append(directory)

    scaling = PhlowerScalingService.from_yaml(_TRAIN_YAML)
    scaling.fit_transform_all(
        interim_data_directories=interim_directories,
        output_base_directory=base / "preprocessed",
        max_process=1,
    )
    preprocessed_directories = [
        base / f"preprocessed/{d.name}" for d in interim_directories
    ]

    setting = PhlowerSetting.read_yaml(_TRAIN_YAML)
    trainer = PhlowerTrainer.from_setting(setting)
    model_directory = base / "model"
    trainer.train(
        output_directory=model_directory,
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
    )
    return _TrainedModel(
        model_directory=model_directory,
        interim_directories=interim_directories,
        preprocessed_directories=preprocessed_directories,
        scaling_yaml=base / "preprocessed/preprocess.yml",
    )


@pytest.fixture(scope="module")
def trained_model_with_noise(
    trained_model: _TrainedModel, tmp_path_factory: pytest.TempPathFactory
) -> _TrainedModel:
    """The same model trained with noise on u and p."""
    content = _load_yaml(_TRAIN_YAML)
    content["scaling"] = _load_yaml(trained_model.scaling_yaml)["scaling"]
    content["training"]["noise"] = [
        {"input_variable": "u", "increment_variable": "du", "std": 0.5},
        {"input_variable": "p", "increment_variable": "dp", "std": 0.5},
    ]
    trainer = PhlowerTrainer.from_setting(
        PhlowerSetting.read_dictionary(content)
    )
    model_directory = tmp_path_factory.mktemp("rollout_with_noise") / "model"
    trainer.train(
        output_directory=model_directory,
        train_directories=trained_model.preprocessed_directories,
        validation_directories=trained_model.preprocessed_directories,
    )
    return trained_model._replace(model_directory=model_directory)


def _create_predictor(
    trained_model: _TrainedModel, **rollout
) -> PhlowerPredictor:
    return PhlowerPredictor(
        model_directory=trained_model.model_directory,
        predict_setting=PhlowerPredictorSetting(
            **_prediction_setting(**rollout)
        ),
        scaling_setting=PhlowerSetting.read_yaml(trained_model.scaling_yaml),
    )


def _create_one_step_predictor(
    trained_model: _TrainedModel,
) -> PhlowerPredictor:
    return PhlowerPredictor(
        model_directory=trained_model.model_directory,
        predict_setting=PhlowerPredictorSetting(
            **_prediction_setting() | {"rollout": None}
        ),
        scaling_setting=PhlowerSetting.read_yaml(trained_model.scaling_yaml),
    )


def test__rollout_from_setting_files(trained_model: _TrainedModel):
    predictor = PhlowerPredictor.from_pathes(
        model_directory=trained_model.model_directory,
        predict_setting_yaml=_PREDICT_YAML,
        scaling_setting_yaml=trained_model.scaling_yaml,
    )

    results = list(
        predictor.predict(
            trained_model.preprocessed_directories, perform_inverse_scaling=True
        )
    )

    assert len(results) > 0
    for result in results:
        for name in ["nodal_u", "nodal_du", "nodal_p", "nodal_dp"]:
            assert result.prediction_data[name].shape[0] == _N_TIMES - 1


@pytest.mark.parametrize(
    "variables, n_steps",
    [(_U_AND_P, None), (_U_AND_P, 1), (_U_AND_P, 8), (_ONLY_U, None)],
)
def test__rollout(
    variables: list[dict],
    n_steps: int | None,
    trained_model: _TrainedModel,
):
    predictor = _create_predictor(
        trained_model, variables=variables, n_steps=n_steps
    )
    results = predictor.predict(
        trained_model.preprocessed_directories, perform_inverse_scaling=True
    )

    assert len(trained_model.interim_directories) > 0
    assert len(variables) > 0
    for interim, result in zip(
        trained_model.interim_directories, results, strict=True
    ):
        for variable in variables:
            name = variable["state_variable"]
            value = np.load(interim / f"nodal_{name}.npy")
            states = result.prediction_data[f"nodal_{name}"]
            increments = result.prediction_data[f"nodal_d{name}"]

            assert states.shape == (n_steps or _N_TIMES - 1, *value.shape[1:])
            assert increments.shape == states.shape
            # The first step starts from the data, the next ones from
            # the predicted states.
            np.testing.assert_array_almost_equal(
                states[0], value[0] + increments[0], decimal=5
            )
            np.testing.assert_array_almost_equal(
                states[1:], states[:-1] + increments[1:], decimal=5
            )


def test__rollout_takes_other_inputs_from_data(trained_model: _TrainedModel):
    directories = trained_model.preprocessed_directories
    predictor = _create_predictor(trained_model, variables=_ONLY_U)
    actual = list(predictor.predict(directories, perform_inverse_scaling=True))
    one_step = _create_one_step_predictor(trained_model)
    desired = list(one_step.predict(directories, perform_inverse_scaling=True))

    # NOTE: p is taken from the data at every step,
    # so dp is the same as without rollout.
    assert len(actual) > 0
    for actual_result, desired_result in zip(actual, desired, strict=True):
        assert "nodal_p" not in actual_result.prediction_data
        np.testing.assert_array_almost_equal(
            actual_result.prediction_data["nodal_dp"],
            desired_result.prediction_data["nodal_dp"],
            decimal=5,
        )


def test__rollout_rescales_with_noise(
    trained_model_with_noise: _TrainedModel,
):
    directories = trained_model_with_noise.preprocessed_directories
    predictor = _create_predictor(trained_model_with_noise, variables=_ONLY_U)
    actual = list(predictor.predict(directories, perform_inverse_scaling=True))
    one_step = _create_one_step_predictor(trained_model_with_noise)
    desired = list(one_step.predict(directories, perform_inverse_scaling=True))

    # NOTE: The first step starts from the data and p is taken from
    # the data at every step, so du at the first step and dp are the same
    # as without rollout, which rescales the whole batch at once.
    assert len(actual) > 0
    for actual_result, desired_result in zip(actual, desired, strict=True):
        np.testing.assert_array_almost_equal(
            actual_result.prediction_data["nodal_du"][0],
            desired_result.prediction_data["nodal_du"][0],
            decimal=5,
        )
        np.testing.assert_array_almost_equal(
            actual_result.prediction_data["nodal_dp"],
            desired_result.prediction_data["nodal_dp"],
            decimal=5,
        )


def test__rollout_with_time_loop_network(
    trained_model: _TrainedModel, tmp_path: pathlib.Path
):
    # NOTE: The time loop adds no parameters,
    # so the network with it loads the same weights.
    model_directory = tmp_path / "model"
    shutil.copytree(trained_model.model_directory, model_directory)
    saved = _load_yaml(model_directory / "model.yml")
    saved["model"]["network"]["time_series_length"] = -1
    with open(model_directory / "model.yml", "w") as fw:
        yaml.safe_dump(saved, fw)
    predictor = _create_predictor(
        trained_model._replace(model_directory=model_directory)
    )
    directories = trained_model.preprocessed_directories
    actual = list(predictor.predict(directories, perform_inverse_scaling=True))
    desired = list(
        _create_predictor(trained_model).predict(
            directories, perform_inverse_scaling=True
        )
    )

    assert len(actual) > 0
    for actual_result, desired_result in zip(actual, desired, strict=True):
        for name in ["nodal_u", "nodal_du", "nodal_p", "nodal_dp"]:
            np.testing.assert_array_almost_equal(
                actual_result.prediction_data[name],
                desired_result.prediction_data[name],
                decimal=5,
            )


def test__rollout_without_inverse_scaling(trained_model: _TrainedModel):
    predictor = _create_predictor(trained_model, n_steps=3)

    result = next(predictor.predict(trained_model.preprocessed_directories))

    for name in ["u", "du", "p", "dp"]:
        assert result.prediction_data[name].is_time_series
        assert result.prediction_data[name].shape[0] == 3


_KEY_MAP = {"du": "nodal_du", "dp": "nodal_dp"}


def _scaling_service(
    scaler_names: list[str] | None = None,
) -> PhlowerScalingService:
    names = scaler_names or ["nodal_u", "nodal_du", "nodal_p", "nodal_dp"]
    return PhlowerScalingService(
        PhlowerScalingSetting(
            variable_name_to_scalers=dict.fromkeys(names, "identity")
        )
    )


@pytest.mark.parametrize("time_series_length", [None, 0, -1])
def test__resolve_rollout_variables(time_series_length: int | None):
    actual = _resolve_rollout_variables(
        RolloutSetting(variables=_U_AND_P),
        model_setting=PhlowerModelSetting(
            **_model_with_time_series_length(time_series_length)
        ),
        output_key_map=_KEY_MAP,
        scalers=_scaling_service(),
    )

    assert [tuple(v) for v in actual] == [
        ("u", "nodal_u", "du", "nodal_du"),
        ("p", "nodal_p", "dp", "nodal_dp"),
    ]


def _model_with_time_series_length(time_series_length: int | None) -> dict:
    model = _model_setting()
    model["network"]["time_series_length"] = time_series_length
    return model


def _model_with_two_members() -> dict:
    model = _model_setting()
    model["inputs"][0]["members"].append({"name": "nodal_v"})
    return model


def _model_with_state_as_output() -> dict:
    model = _model_setting()
    model["network"]["outputs"][1]["name"] = "p"
    model["network"]["modules"][1]["output_key"] = "p"
    return model


def _model_with_states_sharing_a_member() -> dict:
    model = _model_setting()
    model["inputs"][1]["members"][0]["name"] = "nodal_u"
    return model


@pytest.mark.parametrize(
    "variables, model, key_map, scaler_names, message",
    [
        (
            _U_AND_P,
            _model_with_time_series_length(3),
            _KEY_MAP,
            None,
            "network iterates over time",
        ),
        (
            [{"state_variable": "x", "increment_variable": "du"}],
            _model_setting(),
            _KEY_MAP,
            None,
            "must be an input",
        ),
        (
            _ONLY_U,
            _model_with_two_members(),
            _KEY_MAP,
            None,
            "with only one member",
        ),
        (
            [{"state_variable": "p", "increment_variable": "du"}],
            _model_with_state_as_output(),
            _KEY_MAP,
            None,
            "must not be an output",
        ),
        (
            [{"state_variable": "u", "increment_variable": "dq"}],
            _model_setting(),
            _KEY_MAP,
            None,
            "is not an output",
        ),
        (
            _ONLY_U,
            _model_setting(),
            {"dp": "nodal_dp"},
            None,
            "Set it in output_to_scaler_name",
        ),
        (
            _ONLY_U,
            _model_setting(),
            {"du": "nodal_du", "dp": "nodal_u"},
            None,
            "nodal_u of the state u is already used",
        ),
        (
            _U_AND_P,
            _model_with_states_sharing_a_member(),
            _KEY_MAP,
            None,
            "nodal_u of the state p is already used",
        ),
        (
            _ONLY_U,
            _model_setting(),
            _KEY_MAP,
            ["nodal_u"],
            "Scaler for nodal_du is not found",
        ),
    ],
)
def test__raise_error_for_invalid_rollout_variables(
    variables: list[dict],
    model: dict,
    key_map: dict[str, str],
    scaler_names: list[str] | None,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        _resolve_rollout_variables(
            RolloutSetting(variables=variables),
            model_setting=PhlowerModelSetting(**model),
            output_key_map=key_map,
            scalers=_scaling_service(scaler_names),
        )


@pytest.mark.parametrize(
    "n_steps, time_length, desired",
    [(None, 3, 3), (5, 3, 5), (5, 0, 5)],
)
def test__determine_n_steps(
    n_steps: int | None, time_length: int, desired: int
):
    assert _determine_n_steps(n_steps, time_length) == desired


def test__raise_error_when_n_steps_is_unknown():
    with pytest.raises(ValueError, match="Set n_steps of rollout"):
        _determine_n_steps(None, 0)


def test__raise_error_when_increment_shape_does_not_match(
    trained_model: _TrainedModel,
):
    predictor = _create_predictor(
        trained_model,
        variables=[{"state_variable": "u", "increment_variable": "dp"}],
    )

    with pytest.raises(ValueError, match="Shape of the increment dp"):
        next(predictor.predict(trained_model.preprocessed_directories))


def test__rollout_from_initial_states(trained_model: _TrainedModel):
    directory = trained_model.preprocessed_directories[0]
    predictor = _create_predictor(trained_model, n_steps=_N_TIMES - 1)
    desired = next(predictor.predict([directory], perform_inverse_scaling=True))

    # NOTE: Only the initial states are given, time_slice drops the last step.
    data = {
        name: np.load(directory / f"{name}.npy")[:2]
        for name in ["nodal_u", "nodal_p"]
    }
    actual = next(predictor.predict([data], perform_inverse_scaling=True))

    for name in ["nodal_u", "nodal_p"]:
        np.testing.assert_array_almost_equal(
            actual.prediction_data[name],
            desired.prediction_data[name],
            decimal=5,
        )


def test__raise_error_when_other_time_series_is_too_short(
    trained_model: _TrainedModel,
):
    predictor = _create_predictor(
        trained_model, variables=_ONLY_U, n_steps=_N_TIMES
    )

    with pytest.raises(ValueError, match="Time series input p is shorter"):
        next(predictor.predict(trained_model.preprocessed_directories))


def test__rollout_needs_scaling(trained_model: _TrainedModel):
    with pytest.raises(ValueError, match="rollout needs the scaling"):
        PhlowerPredictor.from_pathes(
            model_directory=trained_model.model_directory,
            predict_setting_yaml=_PREDICT_YAML,
        )
