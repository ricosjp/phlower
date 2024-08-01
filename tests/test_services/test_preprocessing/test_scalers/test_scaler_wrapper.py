import numpy as np
import pytest
import scipy.sparse as sp

from phlower.services.preprocessing._scalers import (
    PhlowerScalerWrapper,
    scale_functions,
)


@pytest.fixture
def scalers_name() -> str:
    scalers_name = [
        "identity",
        "standardize",
        "std_scale",
        "isoam_scale",
        "min_max",
        "max_abs_powered",
    ]
    return scalers_name


@pytest.fixture
def sample_data() -> np.ndarray:
    return np.random.rand(100, 3)


@pytest.fixture
def sample_sparse_data() -> sp.csc_matrix:
    return sp.csc_matrix(np.random.rand(100, 3))


@pytest.mark.parametrize(
    "scaler_name, cls",
    [
        ("identity", scale_functions.IdentityScaler),
        ("standardize", scale_functions.StandardScaler),
        ("std_scale", scale_functions.StandardScaler),
        ("isoam_scale", scale_functions.IsoAMScaler),
        ("min_max", scale_functions.MinMaxScaler),
        ("max_abs_powered", scale_functions.MaxAbsPoweredScaler),
    ],
)
def test__initialized_class(scaler_name, cls):
    kwards = {}
    if scaler_name == "isoam_scale":
        kwards = {"other_components": ["a", "b"]}
    wrapper = PhlowerScalerWrapper(scaler_name, parameters=kwards)
    assert isinstance(wrapper._scaler, cls)


@pytest.mark.parametrize("scaler_name", ["std_scale", "max_abs_powered"])
def test__transform_sparse_data(scaler_name, sample_sparse_data):
    scaler = PhlowerScalerWrapper(scaler_name)
    scaler.partial_fit(sample_sparse_data)
    transformed_data = scaler.transform(sample_sparse_data)

    assert transformed_data.shape == sample_sparse_data.shape
    assert isinstance(transformed_data, sp.coo_matrix)


@pytest.mark.parametrize("scaler_name", ["std_scale", "standardize", "min_max"])
def test__get_dumped_data(scaler_name, sample_data):
    scaler = PhlowerScalerWrapper(scaler_name)
    scaler.partial_fit(sample_data)

    n_samples_seen_ = sample_data.size
    dumped_data = scaler.get_dumped_data()
    assert dumped_data.method == scaler.method_name
    assert dumped_data.component_wise == scaler.componentwise
    assert dumped_data.parameters["n_samples_seen_"] == n_samples_seen_
