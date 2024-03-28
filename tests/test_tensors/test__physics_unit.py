import pytest

from phlower.tensors._units import PhysicsUnitTensor


@pytest.mark.parametrize("inputs", [
    [0, 2, 0, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 0, 0]
])
def test__initialize(inputs):
    unit_tensor = PhysicsUnitTensor.create(inputs)


@pytest.mark.parametrize("unit1, unit2", [
    (
        [0, 2, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0],    
    ),
    (
        [1, 2, 0, 2, 0, 0, 0],
        [1, 2, 0, 2, 0, 0, 0],
    )
])
def test__add(unit1, unit2):
    unit1 = PhysicsUnitTensor.create(unit1)
    unit2 = PhysicsUnitTensor.create(unit2)

    assert unit1 == (unit1 + unit2)
