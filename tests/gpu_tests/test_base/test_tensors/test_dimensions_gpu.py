from collections.abc import Callable

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from phlower._base import PhlowerDimensionTensor
from phlower.utils.enums import PhysicalDimensionSymbolType


@st.composite
def random_phlower_dimension_tensor(
    draw: Callable,
) -> PhlowerDimensionTensor:
    dimensions = draw(
        st.lists(
            elements=st.floats(allow_nan=False, allow_infinity=False),
            min_size=len(PhysicalDimensionSymbolType),
            max_size=len(PhysicalDimensionSymbolType),
        )
    )

    return PhlowerDimensionTensor.from_list(dimensions).to(device="cuda:0")


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div],
)
@given(
    st.lists(
        elements=st.floats(width=32, allow_nan=False, allow_infinity=False),
        min_size=len(PhysicalDimensionSymbolType),
        max_size=len(PhysicalDimensionSymbolType),
    )
)
@settings(deadline=None)
def test__op_dimension_tensor_with_same_values_on_gpu(
    op: Callable[
        [PhlowerDimensionTensor, PhlowerDimensionTensor], PhlowerDimensionTensor
    ],
    dimensions: list[float],
):
    a = PhlowerDimensionTensor.from_list(dimensions).to("cuda:0")
    b = PhlowerDimensionTensor.from_list(dimensions).to("cuda:0")
    c = op(a, b)

    cpu_c = op(a.to("cpu"), b.to("cpu"))
    assert c == cpu_c.to("cuda:0")
    assert c.device == torch.device("cuda:0")


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "op",
    [torch.mul, torch.div],
)
@given(random_phlower_dimension_tensor(), st.floats())
def test__op_dimension_tensor_and_float_on_gpu(
    op: Callable[
        [PhlowerDimensionTensor, PhlowerDimensionTensor], PhlowerDimensionTensor
    ],
    a: PhlowerDimensionTensor,
    b: float,
):
    c = op(a, b)

    cpu_c = op(a.to("cpu"), b)
    assert c == cpu_c.to("cuda:0")
    assert c.device == torch.device("cuda:0")


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div],
)
@given(st.floats())
def test__op_zero_dimension_tensor_on_gpu(
    op: Callable[
        [PhlowerDimensionTensor, PhlowerDimensionTensor], PhlowerDimensionTensor
    ],
    x: float,
):
    a = PhlowerDimensionTensor().to("cuda:0")
    c = op(a, x)
    assert c.device == torch.device("cuda:0")
    assert c.is_dimensionless
