"""
.. _first:

phlower is a deep learning framework based on PyTorch Ignite especially for physical phenomenon such as fluid dynamics.


Phlower Tensor Operation with physical dimension
----------------------------------------------------

Usually, physics values have physics dimensions. Phlower can consider it.

"""

###################################################################################################
# Let's check what kind of items are treated as physics dimension
# There are many possible choices of base physical dimensions.
# Following to the SI standard, physical dimensions and corresponding dimension symbols are shown blelow.
#
# time (T), length (L), mass (M), electric current (I), absolute temperature (Θ), amount of substance (N) and luminous intensity (J).
from phlower_tensor.utils.enums import PhysicalDimensionSymbolType

print(f"{[str(v) for v in PhysicalDimensionSymbolType]}")


###################################################################################################
# Create PhlowerTensor which is tensor object with physical dimension.
# There are several way to create PhlowerTensor, the most simple way is shown.
# PhlowerTensor is a wrapper of torch.Tensor. Call `print` to check values and dimensions.

import torch

from phlower_tensor import phlower_tensor

sample_velocity = torch.rand(3, 4)
dimension = {"L": 1, "T": -1}
physical_tensor = phlower_tensor(sample_velocity, dimension=dimension)

print(physical_tensor)


###################################################################################################
# Let's calculate square of velocity.
# You can find that physical dimension is also converted to new dimension.

square_velocity = torch.pow(physical_tensor, 2)
print(square_velocity)


###################################################################################################
# Create PhlowerTensor without physical dimension if you do not pass information of dimension to phlower_tensor.
# Raise error when it comes to calculate PhlowerTensor with a physical dimension and that without it.

try:
    one_physical_tensor = phlower_tensor(
        torch.rand(3, 4), dimension={"L": 1, "T": -1}
    )
    another_physical_tensor = phlower_tensor(torch.rand(3, 4))
    _ = another_physical_tensor + one_physical_tensor
except Exception as ex:
    print(ex)


###################################################################################################
# Some calculation operations are not allowed when physics dimension value is not the same,

try:
    one_physical_tensor = phlower_tensor(
        torch.rand(3, 4), dimension={"L": 1, "T": -1}
    )
    another_physical_tensor = phlower_tensor(
        torch.rand(3, 4), dimension={"L": 1}
    )
    _ = another_physical_tensor + one_physical_tensor
except Exception as ex:
    print(ex)


###################################################################################################
# Some calculation operations are allowed even when physics dimension value is not the same,

one_physical_tensor = phlower_tensor(
    torch.rand(3, 4), dimension={"L": 1, "T": -1}
)
another_physical_tensor = phlower_tensor(
    torch.rand(3, 4), dimension={"Theta": 1}
)
print(another_physical_tensor * one_physical_tensor)
