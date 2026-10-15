import numpy as np
import pytest
import scipy.sparse as sp

from phlower.services.preprocessing._scalers import (
    PhlowerScalerWrapper,
    scale_functions,
)
from phlower.settings import PhlowerScalingSetting
from phlower.utils.typing import DenseArrayType, SparseArrayType


@pytest.fixture
def sample_data() -> np.ndarray:
    return np.random.rand(100, 3)


@pytest.fixture
def sample_sparse_data() -> sp.csc_matrix:
    return sp.coo_matrix(np.random.rand(100, 3))


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
def test__initialized_class(
    scaler_name: str, cls: scale_functions.IPhlowerScaler
):
    kwards = {}
    if scaler_name == "isoam_scale":
        kwards = {"other_components": ["a", "b"]}
    wrapper = PhlowerScalerWrapper(scaler_name, parameters=kwards)
    assert isinstance(wrapper._scaler, cls)


@pytest.mark.parametrize("scaler_name", ["std_scale", "max_abs_powered"])
def test__transform_sparse_data(
    scaler_name: str, sample_sparse_data: SparseArrayType
):
    scaler = PhlowerScalerWrapper(scaler_name)
    scaler.partial_fit(sample_sparse_data)
    transformed_data = scaler.transform(sample_sparse_data)

    assert transformed_data.shape == sample_sparse_data.shape
    assert isinstance(transformed_data.to_numpy(), sp.coo_matrix)


@pytest.mark.parametrize("scaler_name", ["std_scale", "standardize", "min_max"])
def test__get_dumped_data(scaler_name: str, sample_data: DenseArrayType):
    scaler = PhlowerScalerWrapper(scaler_name)
    scaler.partial_fit(sample_data)

    n_samples_seen_ = sample_data.size
    dumped_data = scaler.get_dumped_data()
    assert dumped_data.method == scaler.method_name
    assert dumped_data.component_wise == scaler.componentwise
    assert dumped_data.parameters["n_samples_seen_"] == n_samples_seen_


@pytest.mark.parametrize(
    "shape", [(100, 3, 1), (100, 2, 1), (100, 3, 2), (5, 100, 3, 1)]
)
def test__scale_each_vector_component(shape: tuple[int]):
    n_components = shape[-2]
    scales = 10.0 ** np.arange(n_components)[:, None]
    data = np.random.randn(*shape) * scales

    scaler = PhlowerScalerWrapper(
        "standardize", componentwise=True, component_axis=-2
    )
    scaler.partial_fit(data)
    transformed = scaler.transform(data).to_numpy()

    assert transformed.shape == data.shape
    components = np.reshape(
        np.moveaxis(transformed, -2, -1), (-1, n_components)
    )
    np.testing.assert_array_almost_equal(components.mean(axis=0), 0.0)
    np.testing.assert_array_almost_equal(components.std(axis=0), 1.0)

    inversed = scaler.inverse_transform(transformed).to_numpy()
    np.testing.assert_array_almost_equal(inversed, data)


@pytest.mark.parametrize("kwargs", [{}, {"component_axis": -1}])
def test__vector_is_scaled_with_one_scaler_by_default(kwargs: dict):
    data = np.random.randn(100, 3, 1) * np.array([1.0, 10.0, 100.0])[:, None]

    scaler = PhlowerScalerWrapper("standardize", componentwise=True, **kwargs)
    scaler.partial_fit(data)

    assert scaler._scaler.mean_.shape == (1,)


def test__component_axis_scales_like_componentwise():
    data = np.random.rand(100, 10)

    scaler = PhlowerScalerWrapper("standardize", componentwise=True)
    scaler.partial_fit(data)
    axis_scaler = PhlowerScalerWrapper(
        "standardize", componentwise=True, component_axis=-2
    )
    axis_scaler.partial_fit(data[..., None])

    assert scaler._scaler.mean_.shape == (10,)
    assert axis_scaler._scaler.mean_.shape == (10,)
    np.testing.assert_array_almost_equal(
        axis_scaler._scaler.mean_, scaler._scaler.mean_
    )
    np.testing.assert_array_almost_equal(
        axis_scaler._scaler.var_, scaler._scaler.var_
    )
    np.testing.assert_array_almost_equal(
        axis_scaler.transform(data[..., None]).to_numpy()[..., 0],
        scaler.transform(data).to_numpy(),
    )


def test__dumped_data_has_component_axis():
    scaler = PhlowerScalerWrapper(
        "standardize", componentwise=True, component_axis=-2
    )
    scaler.partial_fit(np.random.rand(100, 3, 1))

    assert scaler.get_dumped_data().component_axis == -2


def test__component_axis_is_passed_from_setting():
    setting = PhlowerScalingSetting(
        variable_name_to_scalers={
            "nodal_u": {
                "method": "standardize",
                "component_wise": True,
                "component_axis": -2,
            }
        }
    )

    scaler = PhlowerScalerWrapper.from_setting(setting.resolve_scalers()[0])

    assert scaler.componentwise
    assert scaler.component_axis == -2


def test__raise_error_when_no_component_axis():
    scaler = PhlowerScalerWrapper(
        "standardize", componentwise=True, component_axis=-2
    )
    with pytest.raises(ValueError, match="no axis"):
        scaler.partial_fit(np.random.rand(100))
