from collections.abc import Callable

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as extra_np
from phlower import PhlowerTensor, phlower_tensor
from phlower.nn._functionals import squeeze
from phlower.utils.enums import PhysicalDimensionSymbolType
from phlower.utils.exceptions import DimensionIncompatibleError


@st.composite
def random_phlower_tensors_with_same_dimension_and_shape(
    draw: Callable,
    shape: tuple[int] | st.SearchStrategy[int],
    zero_dimension: bool = False,
) -> PhlowerTensor:
    _shape = draw(shape)

    array1 = draw(
        extra_np.arrays(
            dtype=np.dtypes.Float32DType(),
            shape=_shape,
        )
    )
    if zero_dimension:
        dimensions = {}
    else:
        dimensions = draw(
            st.lists(
                elements=st.floats(
                    allow_nan=False, allow_infinity=False, width=16
                ),
                min_size=len(PhysicalDimensionSymbolType),
                max_size=len(PhysicalDimensionSymbolType),
            )
        )
        # To avoid zero dimension
        dimensions = [d + 1e-5 for d in dimensions]

    return phlower_tensor(torch.from_numpy(array1), dimension=dimensions)


@pytest.mark.parametrize("func", [torch.sin])
@given(
    value=random_phlower_tensors_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
        zero_dimension=True,
    )
)
def test__torch_functions(value: PhlowerTensor, func: Callable):
    actual = func(value)

    desired = torch.from_numpy(np.sin(value.numpy()))
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )


@pytest.mark.parametrize("func", [torch.sin])
@given(
    value=random_phlower_tensors_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
        zero_dimension=False,
    )
)
def test__torch_functions_raise_error(value: PhlowerTensor, func: Callable):
    with pytest.raises(
        DimensionIncompatibleError, match="Should be dimensionless to apply"
    ):
        _ = func(value)


@given(
    value=random_phlower_tensors_with_same_dimension_and_shape(
        shape=st.lists(
            st.integers(min_value=1, max_value=10), min_size=1, max_size=5
        ),
        zero_dimension=False,
    )
)
def test__torch_where(value: PhlowerTensor):
    scalar = phlower_tensor(torch.tensor(0.5), dimension=value.dimension)
    actual: PhlowerTensor = torch.where(value > scalar, value, 0.0)

    np_value = value.numpy()
    desired = torch.from_numpy(np.where(np_value > 0.5, np_value, 0.0))
    np.testing.assert_array_almost_equal(
        actual.numpy(), desired.numpy(), decimal=5
    )
    np.testing.assert_array_almost_equal(
        actual.dimension.numpy(),
        value.dimension.numpy(),
        decimal=5,
    )


@pytest.mark.parametrize(
    "dim, shape, desired_shape",
    [
        (None, (3, 4, 1, 5), (3, 4, 5)),
        (None, (3, 1, 1, 5), (3, 5)),
        (0, (1, 3, 1, 4, 5), (3, 1, 4, 5)),
        (1, (3, 1, 1, 1, 5), (3, 1, 1, 5)),
        (2, (3, 4, 1, 5), (3, 4, 5)),
    ],
)
def test__torch_squeeze(
    dim: int | None, shape: tuple[int, ...], desired_shape: tuple[int, ...]
):
    value = phlower_tensor(
        torch.rand(shape),
        dimension=[1.0 for _ in range(len(PhysicalDimensionSymbolType))],
    )
    if dim is not None:
        actual: PhlowerTensor = squeeze(value, dim=dim)
    else:
        actual: PhlowerTensor = squeeze(value)

    desired = torch.rand(desired_shape)
    assert actual.shape == desired.shape
    np.testing.assert_array_almost_equal(
        actual.dimension.numpy(), value.dimension.numpy(), decimal=5
    )
