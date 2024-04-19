import pytest
import torch

from phlower import PhlowerDimensionTensor
from phlower.utils.exceptions import DimensionIncompatibleError


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


@pytest.mark.parametrize(
    "unit",
    [
        [0, 2, 0, 0, 0, 0, 0],
        [1, 2, 0, 2, 0, 0, 0],
    ],
)
@pytest.mark.parametrize("dim", [0, 2])
def test__cat(unit, dim):
    unit1 = PhlowerDimensionTensor.from_list(unit)
    unit2 = PhlowerDimensionTensor.from_list(unit)

    assert unit1 == torch.cat([unit1, unit2], dim=dim)


@pytest.mark.parametrize(
    "unit1, unit2",
    [
        (
            [0, 2, 0, 0, 0, 0, 0],
            [0, 2, 1, 0, 0, 0, 0],
        ),
        (
            [1, 2, 0, 2, 0, 0, 0],
            [1, 2, -1, 2, 0, 0, 0],
        ),
    ],
)
def test__cat_raise_dimension_incompatible(unit1, unit2):
    unit1 = PhlowerDimensionTensor.from_list(unit1)
    unit2 = PhlowerDimensionTensor.from_list(unit2)

    with pytest.raises(DimensionIncompatibleError):
        torch.cat([unit1, unit2], dim=0)
