import numpy as np
import pytest

from phlower.services.preprocessing._scalers.scale_functions import IsoAMScaler


@pytest.mark.parametrize(
    "name, kwards",
    [
        ("isoam_scale", {"other_components": ["c"]}),
        ("isoam_scale", {"other_components": ["a", "b"]}),
    ],
)
def test__create(name, kwards):
    scaler = IsoAMScaler.create(name, **kwards)
    assert scaler.other_components == kwards["other_components"]


@pytest.mark.parametrize(
    "name, kwards",
    [
        ("identity", {"other_components": ["c"]}),
        ("maxabs_scale", {"other_components": ["a", "b"]}),
    ],
)
def test__cannot_create(name, kwards):
    with pytest.raises(NotImplementedError):
        IsoAMScaler.create(name, **kwards)


@pytest.mark.parametrize(
    "n_data_list, other_components",
    [
        ([300], ["a", "b"]),
        ([300, 300], ["a", "b"]),
        ([400, 40, 4], ["a", "b", "c"]),
    ],
)
def test__partial_fit(n_data_list, other_components):
    scaler = IsoAMScaler(other_components=other_components)

    data_list = [np.random.rand(n_data) for n_data in n_data_list]
    for data in data_list:
        scaler.partial_fit(data)

    total_data = np.concatenate(data_list)
    splited = np.split(total_data, sum(n_data_list) / scaler.component_dim)
    desired_var = np.sum([np.sum(arr**2) for arr in splited]) / len(splited)

    np.testing.assert_almost_equal(scaler.var_, desired_var)


@pytest.mark.parametrize("n_data, std", [(100, 5.0), (300, 4.0)])
def test__transform(n_data, std):
    scaler = IsoAMScaler(other_components=["dummy"])
    scaler.std_ = std

    arr = np.random.rand(n_data)
    transformed = scaler.transform(arr)

    np.testing.assert_array_almost_equal(arr / std, transformed)
