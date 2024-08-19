import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as ex_np
from phlower.utils import convert_to_dumped, get_registered_scaler_names
from phlower.utils.enums import PhlowerScalerName


@given(st.sampled_from(PhlowerScalerName))
def test__get_registered_scaler_names(scaler_name_type: PhlowerScalerName):
    names = get_registered_scaler_names()
    assert scaler_name_type.value in names


@given(st.integers())
def test__dumped_int_object(val: int):
    actual = convert_to_dumped(val)
    assert actual == val


@given(ex_np.arrays(dtype=np.float64, shape=(3, 5)))
def test__dumped_numpy_dense_array(val: np.ndarray):
    assume(~np.any(np.isnan(val)))
    actual = convert_to_dumped(val)
    assert actual == val.tolist()


@given(ex_np.from_dtype(np.dtype("float")))
def test__dumped_numpy_scalar(val: np.ndarray):
    actual = convert_to_dumped(val)
    if np.isnan(val):
        assert np.isnan(actual)
    else:
        assert actual == val.item()


@given(st.lists(st.floats()))
def test_dumped_sequence_object(val: list[float]):
    actual = convert_to_dumped(val)
    assert actual == val
