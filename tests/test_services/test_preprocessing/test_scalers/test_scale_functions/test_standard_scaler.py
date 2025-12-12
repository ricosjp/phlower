import numpy as np
import pytest

from phlower.services.preprocessing._scalers.scale_functions import (
    StandardScaler,
)


@pytest.mark.parametrize(
    "name, kwards", [("std_scale", {}), ("standardize", {})]
)
def test__create(name: str, kwards: dict):
    _ = StandardScaler.create(name, **kwards)


@pytest.mark.parametrize(
    "name, kwards",
    [
        ("std_scale", {"with_mean": True}),
        ("standardize", {"with_mean": False}),
    ],
)
def test__invalid_args_create(name: str, kwards: dict):
    with pytest.raises(ValueError):
        _ = StandardScaler.create(name, **kwards)


@pytest.mark.parametrize(
    "name, kwards",
    [("identity", {}), ("isoam_scale", {"other_components": ["a"]})],
)
def test__invalid_name_create(name: str, kwards: dict):
    with pytest.raises(NotImplementedError):
        _ = StandardScaler.create(name, **kwards)


def test__default_value():
    scaler = StandardScaler()
    assert scaler.with_mean
    assert scaler.with_std


def test__fixed_property_and_method():
    scaler = StandardScaler()
    assert not scaler.use_diagonal
    assert not scaler.use_diagonal


@pytest.mark.parametrize(
    "n_nodes, n_feature, means",
    [
        (100000, 5, [1, 2, 3, 4, 5]),
    ],
)
def test__std_scale_values(n_nodes: int, n_feature: int, means: list[int]):
    interim_value = np.random.randn(n_nodes, n_feature) * 2 + np.array([means])

    scaler = StandardScaler.create(name="std_scale")
    scaled_data = scaler.fit_transform(interim_value)

    np.testing.assert_almost_equal(
        interim_value / np.std(interim_value, axis=0), scaled_data, decimal=3
    )

    # After transformed, std is almost 1
    np.testing.assert_almost_equal(np.std(scaled_data, axis=0), 1.0, decimal=3)

    # means are saved as scaled
    data_std = np.std(interim_value, axis=0)
    np.testing.assert_almost_equal(
        np.mean(scaled_data, axis=0), np.array(means) / data_std, decimal=2
    )


@pytest.mark.parametrize(
    "n_nodes, n_feature, means",
    [(100000, 5, [1, 2, 3, 4, 5]), (100000, 3, [10, 21, 3])],
)
def test__standardize_values(n_nodes: int, n_feature: int, means: list[int]):
    interim_value = np.random.randn(n_nodes, n_feature) * 2 + np.array([means])

    scaler = StandardScaler()
    preprocessed = scaler.fit_transform(interim_value)

    np.testing.assert_almost_equal(
        (interim_value - np.mean(interim_value, axis=0))
        / (np.std(interim_value, axis=0)),
        preprocessed,
        decimal=3,
    )

    # After transformed, std is almost 1 and mean is almost zero
    np.testing.assert_almost_equal(np.std(preprocessed, axis=0), 1.0, decimal=3)
    np.testing.assert_almost_equal(
        np.mean(preprocessed, axis=0), np.zeros(n_feature), decimal=3
    )


@pytest.mark.parametrize(
    "name, n_nodes, n_feature",
    [
        ("std_scale", 100000, 5),
        ("standardize", 100000, 3),
    ],
)
def test__retrieve_from_dumped_data(name: str, n_nodes: int, n_feature: int):
    interim_value = np.random.randn(n_nodes, n_feature) * 2

    scaler = StandardScaler.create(name)
    scaler.fit(interim_value)

    dumped = scaler.get_dumped_data()

    new_scaler = StandardScaler.create(name, **dumped)

    state = vars(scaler)
    new_state = vars(new_scaler)

    assert len(state) > 0
    for k, v in state.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_almost_equal(v, new_state[k])
            continue
        if isinstance(v, np.generic):
            np.testing.assert_almost_equal(v, new_state[k])
            continue

        assert v == new_state[k]


@pytest.mark.parametrize("name", ["std_scale", "standardize"])
@pytest.mark.parametrize(
    "n_nodes, n_feature",
    [
        (100000, 5),
        (100000, 3),
    ],
)
def test__inverse_scaling(name: str, n_nodes: int, n_feature: int):
    interim_value = np.random.randn(n_nodes, n_feature) * 2

    scaler = StandardScaler.create(name)
    scaler.fit(interim_value)

    _input = np.random.rand(n_nodes, n_feature)
    _retrieved = scaler.inverse_transform(scaler.transform(_input))

    np.testing.assert_array_almost_equal(_input, _retrieved)
