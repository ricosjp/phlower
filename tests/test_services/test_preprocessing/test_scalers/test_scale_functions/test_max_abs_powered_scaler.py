import numpy as np
import pytest
import scipy.sparse as sp

from phlower.services.preprocessing._scalers.scale_functions import (
    MaxAbsPoweredScaler,
)


@pytest.mark.parametrize("name, kwards", [("max_abs_powered", {})])
def test__create(name, kwards):
    _ = MaxAbsPoweredScaler.create(name, **kwards)


@pytest.mark.parametrize("name", ["isoam_scale", "identity"])
def test__failed_create(name):
    with pytest.raises(NotImplementedError):
        MaxAbsPoweredScaler.create(name)


def test__default_parameter():
    scaler = MaxAbsPoweredScaler()
    assert scaler.power == 1.0


@pytest.mark.parametrize(
    "data, desired",
    [
        (
            [
                np.array(
                    [[1, 10, 100], [2, -20, 200], [3, 30, -300]],
                    dtype=np.float32,
                )
            ],
            [3.0, 30.0, 300.0],
        ),
        (
            [
                np.array(
                    [[1, 10, 100], [2, -20, 200], [3, 30, -300]],
                    dtype=np.float32,
                ),
                np.array(
                    [[12, 0, 0], [2, -0, 20], [3, -10, -2200]],
                    dtype=np.float32,
                ),
            ],
            [12.0, 30.0, 2200.0],
        ),
        (
            [sp.coo_matrix(np.array([[1.0], [0.0], [3.0], [-4.0], [0.0]]))],
            [4.0],
        ),
        (
            [
                sp.coo_matrix(np.array([[1.0], [0.0], [3.0], [-4.0], [0.0]])),
                sp.coo_matrix(
                    np.array([[11.2], [-2.1], [3.0], [-5.0], [-110.0]])
                ),
            ],
            [110.0],
        ),
    ],
)
def test__partial_fit(data, desired):
    scaler = MaxAbsPoweredScaler.create("max_abs_powered")
    for v in data:
        scaler.partial_fit(v)
    np.testing.assert_array_almost_equal(scaler.max_, desired)


@pytest.mark.parametrize(
    "max_data, power, inputs, desired",
    [
        (
            np.array([3, 30, 300], dtype=np.float32),
            1.0,
            np.array(
                [[1, 10, 100], [2, -20, 200], [3, 30, -300]],
                dtype=np.float32,
            ),
            np.array(
                [
                    [1 / 3.0, 10 / 30.0, 100 / 300.0],
                    [2 / 3.0, -20 / 30.0, 200 / 300.0],
                    [3 / 3.0, 30 / 30.0, -300 / 300.0],
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array([3.0, 5.0, 10.0], dtype=np.float32),
            2.0,
            np.array(
                [12.0, -20.0, 100.0],
                dtype=np.float32,
            ),
            np.array(
                [
                    12.0 / (3.0**2.0),
                    -20.0 / (5.0**2.0),
                    100.0 / (10.0**2.0),
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test__transform(max_data, power, inputs, desired):
    scaler = MaxAbsPoweredScaler.create("max_abs_powered", power=power)
    scaler.max_ = max_data

    transformed = scaler.transform(inputs)
    np.testing.assert_array_almost_equal(transformed, desired)


@pytest.mark.parametrize(
    "max_data, power, inputs, desired",
    [
        (
            np.array([4.0], dtype=np.float32),
            1.0,
            sp.coo_matrix(np.array([[1.0], [0.0], [3.0], [-4.0], [0.0]])),
            sp.coo_matrix(
                np.array(
                    [
                        [1.0 / 4.0],
                        [0.0 / 4.0],
                        [3.0 / 4.0],
                        [-4.0 / 4.0],
                        [0.0 / 4.0],
                    ]
                )
            ),
        ),
        (
            np.array([4.0], dtype=np.float32),
            2.2,
            sp.coo_matrix(np.array([[3.0], [7.0]])),
            sp.coo_matrix(
                np.array(
                    [
                        [3.0 / (4.0**2.2)],
                        [7.0 / (4.0**2.2)],
                    ]
                )
            ),
        ),
    ],
)
def test__transform_sparse_array(max_data, power, inputs, desired):
    scaler = MaxAbsPoweredScaler.create("max_abs_powered", power=power)
    scaler.max_ = max_data

    transformed = scaler.transform(inputs)
    assert isinstance(transformed, sp.coo_matrix)
    np.testing.assert_array_almost_equal(
        transformed.toarray(), desired.toarray()
    )


@pytest.mark.parametrize(
    "max_data, power, inputs",
    [
        (
            np.array([3, 30, 300], dtype=np.float32),
            1.0,
            np.array(
                [[1, 10, 100], [2, -20, 200], [3, 30, -300]],
                dtype=np.float32,
            ),
        ),
        (
            np.array([3.0, 5.0, 10.0], dtype=np.float32),
            2.0,
            np.array(
                [12.0, -20.0, 100.0],
                dtype=np.float32,
            ),
        ),
    ],
)
def test__inverse_transform(max_data, power, inputs):
    scaler = MaxAbsPoweredScaler.create("max_abs_powered", power=power)
    scaler.max_ = max_data

    result = scaler.inverse_transform(scaler.transform(inputs))

    np.testing.assert_array_almost_equal(result, inputs, decimal=4)


@pytest.mark.parametrize(
    "max_data, power, inputs",
    [
        (
            np.array([4.0], dtype=np.float32),
            1.0,
            sp.coo_matrix(np.array([[1.0], [0.0], [3.0], [-4.0], [0.0]])),
        ),
        (
            np.array([4.0], dtype=np.float32),
            2.2,
            sp.coo_matrix(np.array([[3.0], [7.0]])),
        ),
    ],
)
def test__inverse_transform_sparse_array(max_data, power, inputs):
    scaler = MaxAbsPoweredScaler.create("max_abs_powered", power=power)
    scaler.max_ = max_data

    result = scaler.inverse_transform(scaler.transform(inputs))
    np.testing.assert_array_almost_equal(inputs.toarray(), result.toarray())


@pytest.mark.parametrize(
    "kwards, data",
    [
        ({}, np.random.rand(3, 5)),
        ({"max_": [0.0, 0.0]}, np.random.rand(3, 5)),
        ({"max_": [1.0, 3.0]}, sp.coo_matrix(np.array([[3.0], [7.0]]))),
    ],
)
def test__raise_error_when_transform(kwards, data):
    scaler = MaxAbsPoweredScaler.create("max_abs_powered")
    for k, v in kwards.items():
        setattr(scaler, k, np.array(v))

    with pytest.raises(ValueError):
        scaler.transform(data)


@pytest.mark.parametrize(
    "name, power, n_nodes, n_feature",
    [
        ("max_abs_powered", 1.0, 100000, 5),
        ("max_abs_powered", 2.0, 100000, 3),
    ],
)
def test__retrieve_from_dumped_data(name, power, n_nodes, n_feature):
    interim_value = np.random.randn(n_nodes, n_feature) * 2

    scaler = MaxAbsPoweredScaler.create(name, power=power)
    scaler.partial_fit(interim_value)
    state = vars(scaler)

    dumped = scaler.get_dumped_data()
    new_scaler = MaxAbsPoweredScaler.create(name, **dumped)
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
