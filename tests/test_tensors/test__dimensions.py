import pytest

from phlower.base.tensors import PhlowerDimensionTensor


@pytest.mark.parametrize(
    "inputs", [[0, 2, 0, 0, 0, 0, 0], [0, 2, 0, 2, 0, 0, 0]]
)
def test__initialize(inputs):
    _ = PhlowerDimensionTensor.from_list(inputs)


@pytest.mark.parametrize(
    "unit1, unit2",
    [
        (
            [0, 2, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0],
        ),
        (
            [1, 2, 0, 2, 0, 0, 0],
            [1, 2, 0, 2, 0, 0, 0],
        ),
    ],
)
def test__add(unit1, unit2):
    unit1 = PhlowerDimensionTensor.from_list(unit1)
    unit2 = PhlowerDimensionTensor.from_list(unit2)

    assert unit1 == (unit1 + unit2)
