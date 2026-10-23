import pathlib
from unittest import mock

import numpy as np
import pytest
import torch
import yaml
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import phlower_tensor_collection

from phlower.data import LumpedTensorData
from phlower.io import PhlowerNumpyFile
from phlower.services.preprocessing import PhlowerScalingService
from phlower.services.utils import (
    NoiseAwareScaler,
    NoneNoiseAwareScaler,
    compute_noise_fractions,
    create_noise_aware_scaler,
)
from phlower.settings import NoiseSetting, PhlowerSetting

_SETTING_YAML = pathlib.Path(__file__).parent / "data/training_noise.yml"
_SHARED_SCALERS_YAML = (
    pathlib.Path(__file__).parent / "data/training_noise_shared_scalers.yml"
)
_N_NODES = 8
_MASKED_NODES = [0, 2, 5]
# NOTE: Values in the yml
_U_STD, _DU_STD = 2.0, 3.0
_NOISE_STD, _FRACTION = 0.5, 0.5


def _setting_dict() -> dict:
    with open(_SETTING_YAML) as fr:
        return yaml.safe_load(fr)


def _create_scaler() -> NoiseAwareScaler:
    return NoiseAwareScaler.from_setting(
        PhlowerSetting.read_yaml(_SETTING_YAML)
    )


def _create_batch(
    is_time_series: bool,
    n_nodes: int = _N_NODES,
    masked_nodes: list[int] = _MASKED_NODES,
    dimension: dict | None = None,
) -> LumpedTensorData:
    shape = (1, n_nodes, 3, 1) if is_time_series else (n_nodes, 3, 1)
    mask = np.zeros((n_nodes, 1))
    mask[masked_nodes] = 1.0
    return LumpedTensorData(
        x_data=phlower_tensor_collection(
            {
                "u": phlower_tensor(
                    np.random.randn(*shape),
                    is_time_series=is_time_series,
                    dimension=dimension,
                )
            }
        ),
        y_data=phlower_tensor_collection(
            {
                "du": phlower_tensor(
                    np.random.randn(*shape),
                    is_time_series=is_time_series,
                    dimension=dimension,
                )
            }
        ),
        field_data=phlower_tensor_collection(
            {"nodal_mask": phlower_tensor(mask)}
        ),
    )


def _ratio(std: float) -> float:
    return 1.0 / np.sqrt(1.0 + _FRACTION * (_NOISE_STD / std) ** 2)


@pytest.mark.parametrize("is_time_series", [False, True])
def test__rescale_without_noise(is_time_series: bool):
    noise_scaler = _create_scaler()
    batch = _create_batch(is_time_series)

    actual = noise_scaler.apply(batch, add_noise=False)

    np.testing.assert_array_almost_equal(
        actual.x_data["u"].numpy(), _ratio(_U_STD) * batch.x_data["u"].numpy()
    )
    np.testing.assert_array_almost_equal(
        actual.y_data["du"].numpy(),
        _ratio(_DU_STD) * batch.y_data["du"].numpy(),
    )
    assert actual.x_data["u"].is_time_series == is_time_series


@pytest.mark.parametrize("is_time_series", [False, True])
def test__add_noise_on_masked_nodes(is_time_series: bool):
    noise_scaler = _create_scaler()
    batch = _create_batch(is_time_series)

    actual = noise_scaler.apply(batch, add_noise=True)

    # The same physical noise is added to u and subtracted from du
    noise_from_u = (
        actual.x_data["u"].numpy() / _ratio(_U_STD) - batch.x_data["u"].numpy()
    ) * _U_STD
    noise_from_du = (
        batch.y_data["du"].numpy()
        - actual.y_data["du"].numpy() / _ratio(_DU_STD)
    ) * _DU_STD
    np.testing.assert_array_almost_equal(noise_from_u, noise_from_du)

    nodes_dim = 1 if is_time_series else 0
    noise_from_u = np.moveaxis(noise_from_u, nodes_dim, 0)
    unmasked = [i for i in range(_N_NODES) if i not in _MASKED_NODES]
    np.testing.assert_array_almost_equal(noise_from_u[unmasked], 0.0)
    assert np.all(noise_from_u[_MASKED_NODES] != 0.0)


def test__noise_has_the_given_std():
    noise_scaler = _create_scaler()
    masked_nodes = list(range(10000))
    batch = _create_batch(
        is_time_series=False, n_nodes=20000, masked_nodes=masked_nodes
    )

    actual = noise_scaler.apply(batch, add_noise=True)

    noise_from_u = (
        actual.x_data["u"].numpy() / _ratio(_U_STD) - batch.x_data["u"].numpy()
    ) * _U_STD
    np.testing.assert_allclose(
        np.std(noise_from_u[masked_nodes]), _NOISE_STD, rtol=0.05
    )


def test__add_noise_to_data_with_physical_dimension():
    noise_scaler = _create_scaler()
    batch = _create_batch(is_time_series=False, dimension={"L": 1, "T": -1})

    actual = noise_scaler.apply(batch, add_noise=True)

    assert actual.x_data["u"].dimension == batch.x_data["u"].dimension
    assert actual.y_data["du"].dimension == batch.y_data["du"].dimension


def test__set_rank_changes_the_noise_once():
    batch = _create_batch(is_time_series=False)
    noise_scaler = _create_scaler()
    rank_0 = noise_scaler.apply(batch, add_noise=True).x_data["u"].numpy()

    noise_scaler = _create_scaler()
    noise_scaler.set_rank(1)
    rank_1_first = noise_scaler.apply(batch, add_noise=True).x_data["u"].numpy()
    noise_scaler.set_rank(1)
    rank_1_second = (
        noise_scaler.apply(batch, add_noise=True).x_data["u"].numpy()
    )

    assert not np.allclose(rank_0, rank_1_first)
    assert not np.allclose(rank_1_first, rank_1_second)


def test__noise_is_different_at_each_call_and_reproducible():
    batch = _create_batch(is_time_series=False)
    noise_scaler = _create_scaler()
    first = noise_scaler.apply(batch, add_noise=True).x_data["u"].numpy()
    second = noise_scaler.apply(batch, add_noise=True).x_data["u"].numpy()
    assert not np.allclose(first, second)

    noise_scaler = _create_scaler()
    np.testing.assert_array_equal(
        noise_scaler.apply(batch, add_noise=True).x_data["u"].numpy(), first
    )


def test__apply_equals_scaling_of_noisy_data():
    u = np.random.randn(_N_NODES, 3, 1)
    du = np.random.randn(_N_NODES, 3, 1)
    physical_noise = np.zeros_like(u)
    physical_noise[_MASKED_NODES] = _NOISE_STD * np.random.randn(
        len(_MASKED_NODES), 3, 1
    )
    scalers = PhlowerScalingService.from_yaml(_SETTING_YAML)
    scaled = scalers.transform({"nodal_u": u, "nodal_du": du}, max_process=1)
    batch = LumpedTensorData(
        x_data=phlower_tensor_collection(
            {"u": phlower_tensor(scaled["nodal_u"].to_numpy())}
        ),
        y_data=phlower_tensor_collection(
            {"du": phlower_tensor(scaled["nodal_du"].to_numpy())}
        ),
        field_data=phlower_tensor_collection({}),
    )

    with mock.patch.object(
        NoiseAwareScaler,
        "_draw_noise",
        return_value=torch.from_numpy(physical_noise),
    ):
        actual = _create_scaler().apply(batch, add_noise=True)

    noisy = scalers.transform(
        {"nodal_u": u + physical_noise, "nodal_du": du - physical_noise},
        max_process=1,
    )
    np.testing.assert_array_almost_equal(
        actual.x_data["u"].numpy(),
        _ratio(_U_STD) * noisy["nodal_u"].to_numpy(),
    )
    np.testing.assert_array_almost_equal(
        actual.y_data["du"].numpy(),
        _ratio(_DU_STD) * noisy["nodal_du"].to_numpy(),
    )


def test__apply_to_inputs():
    noise_scaler = _create_scaler()
    inputs = phlower_tensor_collection(
        {"u": phlower_tensor(torch.rand(_N_NODES, 3, 1))}
    )

    actual = noise_scaler.apply(inputs)

    np.testing.assert_array_almost_equal(
        actual["u"].numpy(), _ratio(_U_STD) * inputs["u"].numpy()
    )


def test__raise_error_when_noise_is_added_to_inputs():
    noise_scaler = _create_scaler()
    inputs = phlower_tensor_collection(
        {"u": phlower_tensor(torch.rand(_N_NODES, 3, 1))}
    )

    with pytest.raises(ValueError, match="Noise is added only to a batch"):
        noise_scaler.apply(inputs, add_noise=True)


def test__remove_training_scaling():
    noise_scaler = _create_scaler()
    outputs = phlower_tensor_collection(
        {"du": phlower_tensor(torch.rand(_N_NODES, 3, 1))}
    )

    actual = noise_scaler.remove_training_scaling(outputs)

    np.testing.assert_array_almost_equal(
        actual["du"].numpy(), outputs["du"].numpy() / _ratio(_DU_STD)
    )


def test__create_noise_aware_scaler():
    setting = PhlowerSetting.read_yaml(_SETTING_YAML)
    without_noise = setting.model_copy(
        update={"training": setting.training.model_copy(update={"noise": []})}
    )
    without_training = setting.model_copy(update={"training": None})

    assert isinstance(create_noise_aware_scaler(setting), NoiseAwareScaler)
    assert isinstance(
        create_noise_aware_scaler(without_noise), NoneNoiseAwareScaler
    )
    assert isinstance(
        create_noise_aware_scaler(without_training), NoneNoiseAwareScaler
    )


def test__none_scaler_changes_nothing():
    noise_scaler = NoneNoiseAwareScaler()
    batch = _create_batch(is_time_series=False)
    noise_scaler.set_rank(1)

    assert noise_scaler.apply(batch, add_noise=True) is batch
    assert noise_scaler.apply(batch.x_data) is batch.x_data
    assert noise_scaler.apply(batch.x_data, add_noise=True) is batch.x_data
    assert noise_scaler.remove_training_scaling(batch.y_data) is batch.y_data


def _setting_without_fraction() -> dict:
    setting = _setting_dict()
    del setting["training"]["noise"][0]["fraction"]
    return setting


def _setting_with_noise_on(input_variable: str) -> dict:
    setting = _setting_dict()
    setting["training"]["noise"][0]["input_variable"] = input_variable
    return setting


def _setting_with_scalers(scalers: dict) -> dict:
    setting = _setting_dict()
    setting["scaling"]["variable_name_to_scalers"] = scalers
    return setting


def _setting_without_scaler(name: str) -> dict:
    setting = _setting_dict()
    del setting["scaling"]["variable_name_to_scalers"][name]
    return setting


def _setting_with_input(input_setting: dict) -> dict:
    setting = _setting_dict()
    setting["model"]["inputs"] = [input_setting]
    return setting


@pytest.mark.parametrize(
    "setting, message",
    [
        (_setting_without_fraction(), "fraction of the noise"),
        (
            _setting_with_scalers(
                {
                    "nodal_u": {"method": "standardize"},
                    "nodal_du": {"method": "std_scale"},
                }
            ),
            "is not fitted",
        ),
        (
            _setting_with_scalers(
                {"nodal_u": "logit_transform", "nodal_du": "identity"}
            ),
            "must be one of .* not logit_transform",
        ),
        (
            _setting_with_input(
                {
                    "name": "u",
                    "members": [{"name": "nodal_u"}, {"name": "nodal_p"}],
                }
            ),
            "must have one member",
        ),
        (
            _setting_with_noise_on("x"),
            "Variable x of the noise is not found",
        ),
        (
            _setting_with_input(
                {
                    "name": "u",
                    "members": [{"name": "nodal_u"}],
                    "random_sampling": {"n_sampled_points": 2},
                }
            ),
            "cannot be used with random sampling",
        ),
        (
            _setting_without_scaler("nodal_du"),
            "Scaler of nodal_du is not found",
        ),
    ],
)
def test__raise_error_for_invalid_noise(setting: dict, message: str):
    with pytest.raises(ValueError, match=message):
        NoiseAwareScaler.from_setting(PhlowerSetting(**setting))


def test__add_noise_without_rescaling_for_other_linear_scalers():
    du_max = 4.0
    noise_scaler = NoiseAwareScaler.from_setting(
        PhlowerSetting(
            **_setting_with_scalers(
                {
                    "nodal_u": "identity",
                    "nodal_du": {
                        "method": "max_abs_powered",
                        "parameters": {"max_": [du_max]},
                    },
                }
            )
        )
    )
    batch = _create_batch(is_time_series=False)

    rescaled = noise_scaler.apply(batch, add_noise=False)
    actual = noise_scaler.apply(batch, add_noise=True)

    np.testing.assert_array_almost_equal(
        rescaled.x_data["u"].numpy(), batch.x_data["u"].numpy()
    )
    np.testing.assert_array_almost_equal(
        rescaled.y_data["du"].numpy(), batch.y_data["du"].numpy()
    )
    # The same physical noise is added to u and subtracted from du
    noise_from_u = actual.x_data["u"].numpy() - batch.x_data["u"].numpy()
    noise_from_du = (
        batch.y_data["du"].numpy() - actual.y_data["du"].numpy()
    ) * du_max
    np.testing.assert_array_almost_equal(noise_from_u, noise_from_du)
    assert np.all(noise_from_u[_MASKED_NODES] != 0.0)


def test__raise_error_without_scaling():
    setting = PhlowerSetting.read_yaml(_SETTING_YAML).model_copy(
        update={"scaling": None}
    )

    with pytest.raises(ValueError, match="Noise needs the scaling setting"):
        NoiseAwareScaler.from_setting(setting)


def test__raise_error_when_increment_is_not_an_output():
    noise_scaler = _create_scaler()
    outputs = phlower_tensor_collection(
        {"dp": phlower_tensor(torch.rand(_N_NODES, 3, 1))}
    )

    with pytest.raises(ValueError, match="Increment du is not an output"):
        noise_scaler.remove_training_scaling(outputs)


def test__raise_error_when_mask_does_not_match():
    noise_scaler = _create_scaler()
    batch = _create_batch(is_time_series=False)
    batch = LumpedTensorData(
        x_data=batch.x_data,
        y_data=batch.y_data,
        field_data=phlower_tensor_collection(
            {"nodal_mask": phlower_tensor(np.ones((_N_NODES + 1, 1)))}
        ),
    )

    with pytest.raises(ValueError, match="does not match the data"):
        noise_scaler.apply(batch, add_noise=True)


def test__items_sharing_a_scaler_have_their_own_factors():
    # NOTE: Noise std of each increment in the yml
    noise_stds = {"du": 0.5, "dv": 2.0}
    noise_scaler = NoiseAwareScaler.from_setting(
        PhlowerSetting.read_yaml(_SHARED_SCALERS_YAML)
    )
    outputs = phlower_tensor_collection(
        {
            "du": phlower_tensor(torch.ones(_N_NODES, 1)),
            "dv": phlower_tensor(torch.ones(_N_NODES, 1)),
        }
    )

    actual = noise_scaler.remove_training_scaling(outputs)

    for name, std in noise_stds.items():
        desired = np.sqrt(1.0 + _FRACTION * (std / _DU_STD) ** 2)
        np.testing.assert_array_almost_equal(actual[name].numpy(), desired)


def test__raise_error_when_mask_field_is_missing():
    noise_scaler = _create_scaler()
    batch = _create_batch(is_time_series=False)
    batch = LumpedTensorData(
        x_data=batch.x_data,
        y_data=batch.y_data,
        field_data=phlower_tensor_collection({}),
    )

    with pytest.raises(ValueError, match="Mask field nodal_mask"):
        noise_scaler.apply(batch, add_noise=True)


def test__compute_noise_fractions(tmp_path: pathlib.Path):
    directories = []
    for i, n_masked in enumerate([2, 5]):
        directory = tmp_path / f"data{i}"
        directory.mkdir()
        mask = np.zeros((10, 1))
        mask[:n_masked] = 1.0
        PhlowerNumpyFile.save(directory, "nodal_mask", mask)
        directories.append(directory)
    settings = [
        NoiseSetting(input_variable="u", increment_variable="du", std=1.0),
        NoiseSetting(
            input_variable="p",
            increment_variable="dp",
            std=1.0,
            mask="nodal_mask",
        ),
    ]

    actual = compute_noise_fractions(settings, directories)

    assert actual == [1.0, 7 / 20]


@pytest.mark.parametrize(
    "mask, message",
    [
        (np.ones((10, 2)), "must have the shape"),
        (np.ones((0, 1)), "is empty"),
        (np.full((10, 1), np.nan), "must be 0 or 1"),
        (np.linspace(-1.0, 1.0, 10)[:, None], "must be 0 or 1"),
    ],
)
def test__raise_error_for_invalid_mask_in_fractions(
    mask: np.ndarray, message: str, tmp_path: pathlib.Path
):
    PhlowerNumpyFile.save(tmp_path, "nodal_mask", mask)
    settings = [
        NoiseSetting(
            input_variable="u",
            increment_variable="du",
            std=1.0,
            mask="nodal_mask",
        )
    ]

    with pytest.raises(ValueError, match=message):
        compute_noise_fractions(settings, [tmp_path])
