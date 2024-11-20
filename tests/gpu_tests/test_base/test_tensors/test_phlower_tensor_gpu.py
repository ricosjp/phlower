from collections.abc import Callable

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as extra_np
from phlower import PhlowerTensor, phlower_tensor
from phlower.utils.enums import PhysicalDimensionSymbolType


@st.composite
def random_gpu_phlower_tensors_with_same_dimension(
    draw: Callable, shape: tuple[int] | st.SearchStrategy[int]
) -> tuple[PhlowerTensor, PhlowerTensor]:
    array1 = draw(
        extra_np.arrays(
            dtype=np.dtypes.Float32DType(),
            shape=shape,
        )
    )
    array2 = draw(
        extra_np.arrays(
            dtype=np.dtypes.Float32DType(),
            shape=shape,
        )
    )

    dimensions = draw(
        st.lists(
            elements=st.floats(allow_nan=False, allow_infinity=False),
            min_size=len(PhysicalDimensionSymbolType),
            max_size=len(PhysicalDimensionSymbolType),
        )
    )

    return (
        phlower_tensor(torch.from_numpy(array1), dimension=dimensions).to(
            device="cuda:0"
        ),
        phlower_tensor(torch.from_numpy(array2), dimension=dimensions).to(
            device="cuda:0"
        ),
    )


@st.composite
def random_gpu_zerodim_phlower_tensor(
    draw: Callable, shape: tuple[int] | st.SearchStrategy[int]
) -> PhlowerTensor:
    array = draw(
        extra_np.arrays(
            dtype=np.dtypes.Float32DType(),
            shape=shape,
        )
    )

    return phlower_tensor(torch.from_numpy(array), dimension={}).to(
        device="cuda:0"
    )


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div],
)
@given(random_gpu_phlower_tensors_with_same_dimension(shape=(3, 4)))
def test__op_tensor_with_dimensions_on_gpu(
    op: Callable[[PhlowerTensor, PhlowerTensor], PhlowerTensor],
    tensors: tuple[PhlowerTensor, PhlowerTensor],
):
    a, b = tensors
    c = op(a, b)
    tc: torch.Tensor = op(a.to_tensor(), b.to_tensor())
    np.testing.assert_almost_equal(c.numpy(), tc.cpu().numpy())

    assert c.device == torch.device("cuda:0")
    assert c.dimension.device == torch.device("cuda:0")


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div],
)
@given(
    random_gpu_zerodim_phlower_tensor(shape=(5, 6)),
    st.floats(),
)
def test__op_tensor_scalar_with_unit_on_gpu(
    op: Callable[[PhlowerTensor, PhlowerTensor], PhlowerTensor],
    a: PhlowerTensor,
    b: float,
):
    c = op(a, b)
    tc: torch.Tensor = op(a.to_tensor(), b)
    np.testing.assert_almost_equal(c.numpy(), tc.cpu().numpy())

    assert c.device == torch.device("cuda:0")
    assert c.dimension.device == torch.device("cuda:0")


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "op",
    [torch.add, torch.sub, torch.mul, torch.div],
)
@given(random_gpu_zerodim_phlower_tensor(shape=(10, 5)))
def test__op_tensor_with_zero_dimensions_unit_on_gpu(
    op: Callable[[PhlowerTensor, PhlowerTensor], PhlowerTensor],
    a: PhlowerTensor,
):
    b = a.clone()
    c = op(a, b)

    tc: torch.Tensor = op(a.to_tensor(), b.to_tensor())
    np.testing.assert_almost_equal(c.numpy(), tc.cpu().numpy())
    assert c.device == torch.device("cuda:0")
    assert c.dimension.device == torch.device("cuda:0")
