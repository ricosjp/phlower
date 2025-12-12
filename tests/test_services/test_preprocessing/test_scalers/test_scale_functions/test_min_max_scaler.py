import numpy as np
import pytest

from phlower.services.preprocessing._scalers.scale_functions import MinMaxScaler


@pytest.mark.parametrize(
    "name, kwards", [("min_max", {}), ("min_max", {"copy": True})]
)
def test__create(name: str, kwards: dict):
    _ = MinMaxScaler.create(name, **kwards)


@pytest.mark.parametrize(
    "name, kwards",
    [("identity", {}), ("isoam_scale", {"other_components": ["a"]})],
)
def test__invalid_name_create(name: str, kwards: dict):
    with pytest.raises(NotImplementedError):
        _ = MinMaxScaler.create(name, **kwards)


def test__fixed_property_and_method():
    scaler = MinMaxScaler()
    assert not scaler.use_diagonal
    assert not scaler.use_diagonal


@pytest.mark.parametrize(
    "name, n_nodes, n_feature",
    [
        ("min_max", 100000, 5),
        ("min_max", 100000, 3),
    ],
)
def test__retrieve_from_dumped_data(name: str, n_nodes: int, n_feature: int):
    interim_value = np.random.randn(n_nodes, n_feature) * 2

    scaler = MinMaxScaler.create(name)
    scaler.fit(interim_value)

    dumped = scaler.get_dumped_data()

    new_scaler = MinMaxScaler.create(name, **dumped)

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


@pytest.mark.parametrize(
    "n_nodes, n_feature",
    [
        (100000, 5),
        (100000, 3),
    ],
)
def test__inverse_scaling(n_nodes: int, n_feature: int):
    interim_value = np.random.randn(n_nodes, n_feature)
    scaler = MinMaxScaler()
    scaler.fit(interim_value)

    _input = np.random.randn(n_nodes, n_feature)
    _retrieved = scaler.inverse_transform(scaler.transform(_input))

    np.testing.assert_array_almost_equal(_input, _retrieved)
