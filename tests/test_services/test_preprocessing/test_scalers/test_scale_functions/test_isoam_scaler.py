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
def test__create(name: str, kwards: dict):
    scaler = IsoAMScaler.create(name, **kwards)
    assert scaler.other_components == kwards["other_components"]


@pytest.mark.parametrize(
    "name, kwards",
    [
        ("identity", {"other_components": ["c"]}),
        ("maxabs_scale", {"other_components": ["a", "b"]}),
    ],
)
def test__cannot_create(name: str, kwards: dict):
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
def test__partial_fit(n_data_list: list[int], other_components: list[str]):
    scaler = IsoAMScaler(other_components=other_components)

    data_list = [np.random.rand(n_data) for n_data in n_data_list]
    for data in data_list:
        scaler.partial_fit(data)

    total_data = np.concatenate(data_list)
    splited = np.split(total_data, sum(n_data_list) / scaler.component_dim)
    desired_var = np.sum([np.sum(arr**2) for arr in splited]) / len(splited)

    np.testing.assert_almost_equal(scaler.var_, desired_var)


@pytest.mark.parametrize("n_data, std", [(100, 5.0), (300, 4.0)])
def test__transform(n_data: int, std: float):
    scaler = IsoAMScaler(other_components=["dummy"])
    scaler.std_ = std

    arr = np.random.rand(n_data)
    transformed = scaler.transform(arr)

    np.testing.assert_array_almost_equal(arr / std, transformed)


@pytest.mark.parametrize(
    "n_data_list, other_components",
    [
        ([300], ["a", "b"]),
        ([300, 300], ["a", "b"]),
        ([400, 40, 4], ["a", "b", "c"]),
    ],
)
def test__retrieve_from_dumped_data(
    n_data_list: list[int], other_components: list[str]
):
    scaler = IsoAMScaler(other_components=other_components)

    data_list = [np.random.rand(n_data) for n_data in n_data_list]
    for data in data_list:
        scaler.partial_fit(data)
    state = vars(scaler)

    dumped = scaler.get_dumped_data()
    new_scaler = IsoAMScaler.create("isoam_scale", **dumped)
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
    "n_data_list, other_components",
    [
        ([300, 300], ["a"]),
        ([400, 40, 4], ["a", "b"]),
    ],
)
@pytest.mark.parametrize("shape", [100, 5, 23])
def test__inverse_transform(
    n_data_list: list[int], other_components: list[str], shape: int
):
    scaler = IsoAMScaler(other_components=other_components)

    data_list = [np.random.rand(n_data) for n_data in n_data_list]
    for data in data_list:
        scaler.partial_fit(data)

    sample = np.random.rand(shape)
    retrieved = scaler.inverse_transform(scaler.transform(sample))

    np.testing.assert_array_almost_equal(sample, retrieved)
