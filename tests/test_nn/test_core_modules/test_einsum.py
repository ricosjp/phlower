import numpy as np
import pytest
import torch
from phlower_tensor import PhlowerTensor
from phlower_tensor.collections import phlower_tensor_collection

from phlower.nn import Einsum


def test__can_call_parameters():
    model = Einsum("identity", "")

    # To check Einsum inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "input_shapes, equation, desired_shape",
    [
        ([(5, 2, 3, 16), (5, 2, 3, 16)], "ijkl,ijkl->il", (5, 16)),
        ([(5, 2, 3, 16), (5, 2, 16)], "ijkl,ijl->ikl", (5, 3, 16)),
        ([(5, 2, 16), (5, 2, 3, 16)], "ijl,ijkl->ikl", (5, 3, 16)),
        (
            [(5, 2, 16), (5, 2, 3, 16), (5, 2, 16, 4)],
            "ijl,ijkl,ijlo->ko",
            (3, 4),
        ),
        ([(5, 2, 3, 16)], "ijkl->il", (5, 16)),
    ],
)
def test__contraction_tensor_shape(
    input_shapes: list[tuple[int]], equation: str, desired_shape: tuple[int]
):
    phlower_tensors = {
        f"phlower_tensor_{i}": PhlowerTensor(
            torch.from_numpy(np.random.rand(*s))
        )
        for i, s in enumerate(input_shapes)
    }
    phlower_tensors = phlower_tensor_collection(phlower_tensors)

    model = Einsum("identity", equation)

    actual = model(phlower_tensors)

    assert actual.shape == desired_shape
