import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st
from phlower_tensor import PhlowerTensor, phlower_tensor

from phlower.services.preprocessing._scalers.scale_functions import (
    PhysicalDataType,
    create_scaler,
)


@given(
    data_type=st.sampled_from(PhysicalDataType),
    rho=st.floats(min_value=0.1, max_value=10.0),
    nu=st.floats(min_value=0.1, max_value=10.0),
    U=st.floats(min_value=0.1, max_value=10.0),
    L=st.floats(min_value=0.1, max_value=10.0),
)
def test__create(
    data_type: PhysicalDataType, rho: float, nu: float, U: float, L: float
):
    kwards = {
        "data_type": data_type.value,
        "rho": rho,
        "nu": nu,
        "U": U,
        "L": L,
    }
    _ = create_scaler("physical_nondimensionalize", **kwards)


def test__fixed_property_and_method():
    scaler = create_scaler(
        "physical_nondimensionalize",
        data_type="velocity",
        rho=1.0,
        nu=1.0,
        U=1.0,
        L=1.0,
    )
    assert not scaler.use_diagonal
    assert not scaler.is_erroneous()
    assert scaler.is_fitted()


@given(
    data_type=st.sampled_from(PhysicalDataType),
    rho=st.floats(min_value=0.1, max_value=10.0),
    nu=st.floats(min_value=0.1, max_value=10.0),
    U=st.floats(min_value=0.1, max_value=10.0),
    L=st.floats(min_value=0.1, max_value=10.0),
)
def test__retrieve_properties(
    data_type: str, rho: float, nu: float, U: float, L: float
):
    scaler = create_scaler(
        "physical_nondimensionalize",
        data_type=data_type,
        rho=rho,
        nu=nu,
        U=U,
        L=L,
    )
    dumped = scaler.get_dumped_data()

    assert dumped["data_type"] == data_type
    assert dumped["rho"] == rho
    assert dumped["nu"] == nu
    assert dumped["U"] == U
    assert dumped["L"] == L

    scaler = create_scaler("physical_nondimensionalize", **dumped)
    assert scaler.get_dumped_data() == dumped


def create_random_physical_data(data_type: PhysicalDataType) -> PhlowerTensor:
    match data_type:
        case PhysicalDataType.velocity:
            return phlower_tensor(torch.rand(100), dimension={"L": 1, "T": -1})
        case PhysicalDataType.pressure:
            return phlower_tensor(
                torch.rand(100), dimension={"M": 1, "L": -1, "T": -2}
            )
        case PhysicalDataType.length:
            return phlower_tensor(torch.rand(100), dimension={"L": 1})
        case PhysicalDataType.pressure_by_density:
            return phlower_tensor(torch.rand(100), dimension={"L": 2, "T": -2})
        case _:
            raise NotImplementedError(f"Invalid data_type: {data_type}")


@given(
    data_type=st.sampled_from(PhysicalDataType),
    rho=st.floats(min_value=0.1, max_value=10.0),
    nu=st.floats(min_value=0.1, max_value=10.0),
    U=st.floats(min_value=0.1, max_value=10.0),
    L=st.floats(min_value=0.1, max_value=10.0),
)
def test__transform_for_all_data_types(
    data_type: PhysicalDataType, rho: float, nu: float, U: float, L: float
):
    # Use phlower_tensor to validate the dimensionlessness

    scaler = create_scaler(
        "physical_nondimensionalize",
        data_type=data_type,
        rho=phlower_tensor(torch.tensor(rho), dimension={"M": 1, "L": -3}),
        nu=phlower_tensor(torch.tensor(nu), dimension={"L": 2, "T": -1}),
        U=phlower_tensor(torch.tensor(U), dimension={"L": 1, "T": -1}),
        L=phlower_tensor(torch.tensor(L), dimension={"L": 1}),
    )

    data = create_random_physical_data(data_type)
    transformed: PhlowerTensor = scaler.transform(data)

    assert transformed.dimension.is_dimensionless


@given(
    data_type=st.sampled_from(PhysicalDataType),
    rho=st.floats(min_value=0.1, max_value=10.0),
    nu=st.floats(min_value=0.1, max_value=10.0),
    U=st.floats(min_value=0.1, max_value=10.0),
    L=st.floats(min_value=0.1, max_value=10.0),
)
def test__inverse_transform(
    data_type: PhysicalDataType, rho: float, nu: float, U: float, L: float
):
    scaler = create_scaler(
        "physical_nondimensionalize",
        data_type=data_type,
        rho=rho,
        nu=nu,
        U=U,
        L=L,
    )

    data = np.random.rand(100)
    transformed = scaler.transform(data)
    inversed = scaler.inverse_transform(transformed)

    np.testing.assert_array_almost_equal(data, inversed)
