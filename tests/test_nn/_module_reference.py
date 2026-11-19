"""NumPy reference computations for module tests."""

import numpy as np
import torch
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def linear(values: FloatArray, layer: torch.nn.Linear) -> FloatArray:
    assert layer.bias is not None
    return (
        values @ layer.weight.detach().numpy().T + layer.bias.detach().numpy()
    )
