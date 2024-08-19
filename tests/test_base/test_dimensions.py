import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from phlower._base import PhysicalDimensions
from phlower.utils.enums import PhysicalDimensionSymbolType
from phlower.utils.exceptions import InvalidDimensionError


@given(
    st.dictionaries(
        st.sampled_from(list(PhysicalDimensionSymbolType.__members__.keys())),
        st.floats(allow_nan=False),
    )
)
def test__equal_when_same_dimension(dict_data: dict[str, float]):
    dimension = PhysicalDimensions(dict_data)
    other = PhysicalDimensions(dict_data)

    assert dimension == other


@pytest.mark.parametrize(
    "dict_data",
    [({"kg": 2, "mm": 3}), ({"m": 2, "hour": 3.2}), ({"mass": None})],
)
def test__failed_when_not_exist_key(dict_data: dict[str, float]):
    with pytest.raises(InvalidDimensionError):
        _ = PhysicalDimensions(dict_data)


@given(
    st.tuples(
        st.dictionaries(
            st.sampled_from(
                list(PhysicalDimensionSymbolType.__members__.keys())
            ),
            st.floats(allow_nan=False, min_value=1),
        ),
        st.dictionaries(
            st.sampled_from(
                list(PhysicalDimensionSymbolType.__members__.keys())
            ),
            st.floats(allow_nan=False, min_value=1),
        ),
    )
)
def test__not_equal_dimension(tuple_dict_data: tuple[dict, dict]):
    dict_data1, dict_data2 = tuple_dict_data
    assume(dict_data1 != dict_data2)

    one = PhysicalDimensions(dict_data1)
    other = PhysicalDimensions(dict_data2)

    assert one != other


def test__default_dimension():
    dimension = PhysicalDimensions({})

    for ptype in PhysicalDimensionSymbolType:
        assert dimension[ptype.name] == 0


@given(
    st.dictionaries(
        st.sampled_from(list(PhysicalDimensionSymbolType.__members__.keys())),
        st.floats(allow_nan=False),
    )
)
def test__to_list(dict_data: dict[str, float]):
    dimension = PhysicalDimensions(dict_data)
    list_data = dimension.to_list()

    for ptype in PhysicalDimensionSymbolType:
        assert list_data[ptype.value] == dict_data.get(ptype.name, 0)
