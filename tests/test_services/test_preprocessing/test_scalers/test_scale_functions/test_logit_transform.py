import numpy as np
import pytest

from phlower.services.preprocessing._scalers.scale_functions import (
    LogitTransformScaler,
)


@pytest.mark.parametrize(
    "name, kwards", [("logit_transform", {}), ("logit_transform", {"some": 11})]
)
def test__create(name: str, kwards: dict):
    LogitTransformScaler.create(name, **kwards)


@pytest.mark.parametrize(
    "name, kwards", [("isoam", {}), ("identities", {"some": 11})]
)
def test__cannot_create(name: str, kwards: dict):
    with pytest.raises(NotImplementedError):
        LogitTransformScaler.create(name, **kwards)


def test__fixed_property_and_method():
    scaler = LogitTransformScaler()
    assert not scaler.use_diagonal
    assert not scaler.is_erroneous()
    assert scaler.is_fitted()


@pytest.mark.parametrize(
    "lower_bound, upper_bound", [(0.001, 0.5), (0.5, 0.9999)]
)
def test__retrieve_properties(lower_bound: float, upper_bound: float):
    scaler = LogitTransformScaler(
        lower_bound=lower_bound, upper_bound=upper_bound
    )
    dumped = scaler.get_dumped_data()

    assert dumped["lower_bound"] == lower_bound
    assert dumped["upper_bound"] == upper_bound

    new_scaler = LogitTransformScaler.create("logit_transform", **dumped)
    new_dumped = new_scaler.get_dumped_data()
    assert new_dumped == dumped


def test__transform():
    scaler = LogitTransformScaler()

    data = np.linspace(0.01, 0.99, num=100)
    transformed = scaler.transform(data)

    expected = np.log(data / (1.0 - data))
    np.testing.assert_array_almost_equal(transformed, expected)


def test__inverse_transform():
    scaler = LogitTransformScaler()

    data = np.linspace(0.01, 0.99, num=100)
    transformed = scaler.transform(data)
    inversed = scaler.inverse_transform(transformed)

    np.testing.assert_array_almost_equal(data, inversed)


def test__transform_with_clip():
    scaler = LogitTransformScaler()

    data = np.linspace(0.0, 1.0, num=100)
    transformed = scaler.transform(data)

    assert not np.any(np.isinf(transformed))

    inversed = scaler.inverse_transform(transformed)
    assert np.all(inversed > 0.0) and np.all(inversed < 1.0)

    np.testing.assert_array_almost_equal(data[1:-1], inversed[1:-1])
