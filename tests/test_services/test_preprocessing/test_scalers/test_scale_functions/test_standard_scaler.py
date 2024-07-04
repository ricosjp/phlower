import numpy as np
import pytest

from phlower.services.preprocessing._scalers.scale_functions import (
    StandardScaler,
)


@pytest.mark.parametrize(
    "name, kwards", [("std_scale", {}), ("standardize", {})]
)
def test__create(name, kwards):
    _ = StandardScaler.create(name, **kwards)


@pytest.mark.parametrize(
    "name, kwards",
    [
        ("std_scale", {"with_mean": True}),
        ("standardize", {"with_mean": False}),
    ],
)
def test__invalid_args_create(name, kwards):
    with pytest.raises(ValueError):
        _ = StandardScaler.create(name, **kwards)


@pytest.mark.parametrize(
    "name, kwards",
    [("identity", {}), ("isoam_scale", {"other_components": ["a"]})],
)
def test__invalid_name_create(name, kwards):
    with pytest.raises(NotImplementedError):
        _ = StandardScaler.create(name, **kwards)


def test__default_value():
    scaler = StandardScaler()
    assert scaler.with_mean
    assert scaler.with_std


@pytest.mark.parametrize(
    "n_nodes, n_feature, means",
    [
        (100000, 5, [1, 2, 3, 4, 5]),
    ],
)
def test__std_scale_values(n_nodes, n_feature, means):
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
def test__standardize_values(n_nodes, n_feature, means):
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
