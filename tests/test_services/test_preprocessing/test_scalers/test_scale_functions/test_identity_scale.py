import numpy as np
import pytest
from phlower.services.preprocessing._scalers.scale_functions import (
    IdentityScaler,
)


@pytest.mark.parametrize(
    "name, kwards", [("identity", {}), ("identity", {"some": 11})]
)
def test__create(name: str, kwards: dict):
    IdentityScaler.create(name, **kwards)


@pytest.mark.parametrize(
    "name, kwards", [("isoam", {}), ("identities", {"some": 11})]
)
def test__cannot_create(name: str, kwards: dict):
    with pytest.raises(NotImplementedError):
        IdentityScaler.create(name, **kwards)


def test__fixed_property_and_method():
    scaler = IdentityScaler()
    assert not scaler.use_diagonal
    assert not scaler.is_erroneous()
    assert scaler.is_fitted()


@pytest.mark.parametrize("shape", [(3, 4), (4, 5), (2, 9)])
def test__transform(shape: tuple[int]):
    scaler = IdentityScaler()

    data = np.random.rand(*shape)
    transformed = scaler.transform(data)

    np.testing.assert_array_almost_equal(data, transformed)


@pytest.mark.parametrize("shape", [(3, 4), (4, 5), (2, 9)])
def test__inverse_transform(shape: tuple[int]):
    scaler = IdentityScaler()

    data = np.random.rand(*shape)
    transformed = scaler.inverse_transform(data)

    np.testing.assert_array_almost_equal(data, transformed)
