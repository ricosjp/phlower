import pytest
import torch

from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import GCN


def test__can_call_parameters():
    model = GCN(nodes=[4, 8], support_name='support')

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "size, is_time_series",
    [
        ((10, 1), False),
        ((10, 16), False),
        ((10, 3, 16), False),
        ((4, 10, 1), True),
        ((4, 10, 16), True),
        ((4, 10, 3, 16), True),
    ],
)
def test__gcn(size, is_time_series):
    phlower_tensor = PhlowerTensor(
        torch.rand(*size), is_time_series=is_time_series)
    phlower_tensors = phlower_tensor_collection({'tensor': phlower_tensor})
    n = phlower_tensor.n_vertices()
    dict_supports = {'support': PhlowerTensor(torch.rand(n, n).to_sparse())}

    model = GCN(
        nodes=[size[-1], size[-1], size[-1]],
        support_name='support', activations=['tanh', 'identity'])

    actual = model(phlower_tensors, supports=dict_supports)

    assert actual.shape == size
    assert actual.is_time_series == is_time_series
