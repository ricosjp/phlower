import numpy as np
import pytest
import torch
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import phlower_tensor_collection

from phlower.nn import Accessor


def test__can_call_parameters():
    model = Accessor("identity")

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "input_shape, activation, index, desired_shape, dim, keepdim",
    [
        ((5, 5, 16), "identity", 0, (5, 16), 0, False),
        ((4, 2, 16), "identity", -1, (2, 16), 0, False),
        ((5, 2, 3, 4), "tanh", 1, (2, 3, 4), 0, False),
        ((3, 2, 1), "relu", 2, (2, 1), 0, False),
        ((2, 3, 4), "identity", 1, (2, 4), 1, False),
        ((2, 3, 4), "identity", 1, (1, 3, 4), 0, True),
        ((2, 3, 4), "identity", 1, (2, 1, 4), 1, True),
        ((2, 3, 4), "identity", 1, (2, 3, 1), -1, True),
        ((2, 3, 4), "identity", [0, 1], (2, 3, 2), -1, True),
    ],
)
def test__accessed_tensor_shape(
    input_shape: tuple[int],
    activation: str,
    index: int | list[int],
    desired_shape: tuple[int],
    dim: int,
    keepdim: bool,
    auto_device: str,
):
    ph_tensor = phlower_tensor(
        torch.from_numpy(np.random.rand(*input_shape)).to(device=auto_device)
    )
    phlower_tensors = phlower_tensor_collection({"tensor": ph_tensor})

    model = Accessor(
        activation=activation, index=index, dim=dim, keepdim=keepdim
    )

    actual = model.forward(phlower_tensors)

    assert actual.shape == desired_shape


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "input_shape, index, dim, keepdim, desired_indices",
    [
        ((5, 5, 16), 0, 0, False, [0, ...]),
        ((4, 2, 16), -1, 0, False, [-1, ...]),
        ((5, 2, 3, 4), 1, 0, False, [1, ...]),
        ((3, 2, 1), 2, 0, False, [2, ...]),
        ((2, 3, 4), 1, 1, False, [slice(None), 1, ...]),
        ((2, 3, 4), 1, 0, True, [slice(1, 2), ...]),
        ((2, 3, 4), 1, 1, True, [slice(None), slice(1, 2), ...]),
        ((2, 3, 4), 1, -1, True, [..., slice(1, 2)]),
        ((2, 3, 4), [0, 1], -1, True, [..., slice(0, 2)]),
    ],
)
def test__accessed_tensor_value(
    input_shape: tuple[int],
    index: int | list[int],
    dim: int,
    keepdim: bool,
    desired_indices: list[int | slice],
    auto_device: str,
):
    ph_tensor = phlower_tensor(
        torch.from_numpy(np.random.rand(*input_shape)).to(device=auto_device)
    )
    phlower_tensors = phlower_tensor_collection({"tensor": ph_tensor})

    model = Accessor(
        activation="identity", index=index, dim=dim, keepdim=keepdim
    )

    actual = model.forward(phlower_tensors)

    desired = phlower_tensors["tensor"].to_tensor()[desired_indices]

    np.testing.assert_array_almost_equal(
        desired.cpu().numpy(),
        actual.to_numpy(),
    )


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "index, dim, keepdim, is_time_series, desired",
    [
        (0, 0, False, False, False),
        (0, 0, False, True, False),
        (0, 0, True, True, True),
        ([0, 1], 0, True, True, True),
        (0, 1, False, True, True),
        (2, 1, True, True, True),
        (2, 1, False, True, True),
    ],
)
def test__time_series_properly_handled(
    index: int,
    dim: int,
    keepdim: bool,
    is_time_series: bool,
    desired: bool,
    auto_device: str,
):
    input_shape = (4, 3, 2)
    ph_tensor = phlower_tensor(
        torch.from_numpy(np.random.rand(*input_shape)).to(device=auto_device),
        is_time_series=is_time_series,
    )
    phlower_tensors = phlower_tensor_collection({"tensor": ph_tensor})

    model = Accessor(
        activation="identity", index=index, dim=dim, keepdim=keepdim
    )
    actual = model.forward(phlower_tensors)

    assert actual.is_time_series == desired
    assert actual.is_voxel is False


@pytest.mark.gpu_test
@pytest.mark.parametrize(
    "index, dim, keepdim, is_time_series, is_voxel, "
    "desired_time_series, desired_voxel",
    [
        (0, 0, False, False, True, False, False),
        (0, 0, True, False, True, False, True),
        (1, 0, False, True, True, False, True),
        (1, 0, True, True, True, True, True),
        ([0, 1], 0, False, True, True, True, True),
        ([0, 1], 1, False, True, True, True, True),
        ([0, 1], 2, False, True, True, True, True),
        ([0, 1], 3, False, True, True, True, True),
        ([0, 1], 4, True, True, True, True, True),
    ],
)
def test__voxel_properly_handled(
    index: int,
    dim: int,
    keepdim: bool,
    is_time_series: bool,
    is_voxel: bool,
    desired_time_series: bool,
    desired_voxel: bool,
    auto_device: str,
):
    input_shape = (4, 3, 5, 2, 2)
    ph_tensor = phlower_tensor(
        torch.from_numpy(np.random.rand(*input_shape)).to(device=auto_device),
        is_time_series=is_time_series,
        is_voxel=is_voxel,
    )
    phlower_tensors = phlower_tensor_collection({"tensor": ph_tensor})

    model = Accessor(
        activation="identity", index=index, dim=dim, keepdim=keepdim
    )
    actual = model.forward(phlower_tensors)

    assert actual.is_time_series is desired_time_series
    assert actual.is_voxel is desired_voxel
