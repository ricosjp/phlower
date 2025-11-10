import numpy as np
import pytest
import torch
from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import Accessor


def test__can_call_parameters():
    model = Accessor("identity")

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


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
):
    phlower_tensor = PhlowerTensor(
        torch.from_numpy(np.random.rand(*input_shape))
    )
    phlower_tensors = phlower_tensor_collection({"tensor": phlower_tensor})

    model = Accessor(
        activation=activation, index=index, dim=dim, keepdim=keepdim
    )

    actual: PhlowerTensor = model(phlower_tensors)

    assert actual.shape == desired_shape

    if activation == "identity":
        if dim == 0:
            if keepdim:
                np.testing.assert_almost_equal(
                    phlower_tensors.to_numpy()["tensor"][index : index + 1],
                    actual.to("cpu").detach().numpy().copy(),
                )
            else:
                np.testing.assert_almost_equal(
                    phlower_tensors.to_numpy()["tensor"][index],
                    actual.to("cpu").detach().numpy().copy(),
                )
        elif dim == 1:
            if keepdim:
                np.testing.assert_almost_equal(
                    phlower_tensors.to_numpy()["tensor"][
                        :, index : index + 1, :
                    ],
                    actual.to("cpu").detach().numpy().copy(),
                )
            else:
                np.testing.assert_almost_equal(
                    phlower_tensors.to_numpy()["tensor"][:, index, :],
                    actual.to("cpu").detach().numpy().copy(),
                )
        elif dim == 2:
            if keepdim:
                if isinstance(index, int):
                    index = [index]
                ans = phlower_tensors.to_numpy()["tensor"][
                    :, :, index[0] : index[0] + 1
                ]
                for i in range(1, len(index)):
                    ans = np.concatenate(
                        (
                            ans,
                            phlower_tensors.to_numpy()["tensor"][
                                :, :, index[i] : index[i] + 1
                            ],
                        ),
                        axis=-1,
                    )

                np.testing.assert_almost_equal(
                    ans,
                    actual.to("cpu").detach().numpy().copy(),
                )
            else:
                np.testing.assert_almost_equal(
                    phlower_tensors.to_numpy()["tensor"][:, :, index],
                    actual.to("cpu").detach().numpy().copy(),
                )
