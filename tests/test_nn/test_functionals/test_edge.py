import numpy as np
import torch
from phlower_tensor import phlower_tensor

from phlower.nn._functionals import extract_edge_indices


def test__uncoalesced_support_gives_row_major_edges():
    # path graph 0 - 1 - 2 with both edge directions and self-loops,
    # entries in random order:
    # [[1, 1, 0],
    #  [1, 1, 1],
    #  [0, 1, 1]]
    indices = torch.tensor([[2, 1, 0, 1, 2, 0, 1], [1, 2, 0, 1, 2, 1, 0]])
    support = phlower_tensor(
        torch.sparse_coo_tensor(indices, torch.ones(7), size=(3, 3))
    )
    assert not support.to_tensor().is_coalesced()

    receiver, sender = extract_edge_indices(support)

    # row-major edge order with self-loops removed:
    # (0, 1), (1, 0), (1, 2), (2, 1)
    np.testing.assert_array_equal(receiver.numpy(), [0, 1, 1, 2])
    np.testing.assert_array_equal(sender.numpy(), [1, 0, 2, 1])
