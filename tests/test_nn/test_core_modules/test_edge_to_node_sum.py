import numpy as np
import pytest
import torch
from phlower_tensor import PhlowerTensor, phlower_tensor
from phlower_tensor.collections import phlower_tensor_collection
from phlower_tensor.functionals import to_batch

from phlower.nn import EdgeDifference, EdgeGather, EdgeToNodeSum
from phlower.settings._module_settings import EdgeToNodeSumSetting


def _random_support(
    n_nodes: int, with_self_loops: bool = True
) -> PhlowerTensor:
    adjacency = np.triu(np.random.rand(n_nodes, n_nodes) > 0.5, k=1)
    adjacency = adjacency + adjacency.T
    if with_self_loops:
        adjacency = adjacency + np.eye(n_nodes, dtype=bool)
    return phlower_tensor(
        torch.from_numpy(adjacency.astype(np.float32)).to_sparse()
    )


def _adjacency_without_self_loops(support: PhlowerTensor) -> np.ndarray:
    dense = support.to_tensor().to_dense().numpy().copy()
    np.fill_diagonal(dense, 0.0)
    return dense


def _path_graph_support() -> PhlowerTensor:
    # path graph 0 - 1 - 2 with both edge directions and self-loops:
    # [[1, 1, 0],
    #  [1, 1, 1],
    #  [0, 1, 1]]
    indices = torch.tensor([[0, 0, 1, 1, 1, 2, 2], [0, 1, 0, 1, 2, 1, 2]])
    return phlower_tensor(
        torch.sparse_coo_tensor(indices, torch.ones(7), size=(3, 3)).coalesce()
    )


def test__can_call_parameters():
    model = EdgeToNodeSum(support_name="support")

    # To check EdgeToNodeSum inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize("support_name", ["aaa", "support1"])
def test__can_pass_parameters_via_setting(support_name: str):
    setting = EdgeToNodeSumSetting(support_name=support_name)
    model = EdgeToNodeSum.from_setting(setting)

    assert model._support_name == support_name


def test__sum_on_path_graph():
    support = _path_graph_support()
    # coalesced edge order: (0, 1), (1, 0), (1, 2), (2, 1)
    edge = np.array(
        [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0], [1000.0, 2000.0]],
        dtype=np.float32,
    )
    inputs = phlower_tensor_collection(
        {"edge": phlower_tensor(torch.from_numpy(edge))}
    )

    model = EdgeToNodeSum(support_name="support")
    actual = model(inputs, field_data={"support": support})

    desired = np.array([edge[0], edge[1] + edge[2], edge[3]])
    np.testing.assert_array_almost_equal(actual.to_numpy(), desired)


def test__batch_correct():
    # a batched support is block-diagonal, so samples must not interact
    supports = [_path_graph_support(), _random_support(5)]
    edges = [
        torch.rand(len(support.values()) - support.shape[0], 2)
        for support in supports
    ]
    batched_support, _ = to_batch(supports)
    batched_inputs = phlower_tensor_collection(
        {"edge": phlower_tensor(torch.concatenate(edges))}
    )

    model = EdgeToNodeSum(support_name="support")
    actual = model(batched_inputs, field_data={"support": batched_support})

    desired = np.concatenate(
        [
            model(
                phlower_tensor_collection({"edge": phlower_tensor(edge)}),
                field_data={"support": support},
            ).to_numpy()
            for support, edge in zip(supports, edges, strict=True)
        ]
    )
    np.testing.assert_array_almost_equal(actual.to_numpy(), desired)


@pytest.mark.parametrize("size", [(3,), (3, 4), (3, 3, 4)])
def test__keeps_trailing_shape(size: tuple[int]):
    n_nodes = 10
    support = _random_support(n_nodes)
    n_edges = len(support.values()) - n_nodes
    inputs = phlower_tensor_collection(
        {"edge": phlower_tensor(torch.rand(n_edges, *size))}
    )

    model = EdgeToNodeSum(support_name="support")
    actual = model(inputs, field_data={"support": support})

    assert actual.shape == (n_nodes, *size)


def test__sum_of_differences_matches_dense_formula():
    n_nodes = 10
    n_feature = 3
    support = _random_support(n_nodes)
    adjacency = _adjacency_without_self_loops(support)
    h = np.random.rand(n_nodes, n_feature).astype(np.float32)
    inputs = phlower_tensor_collection(
        {"h": phlower_tensor(torch.from_numpy(h))}
    )
    field_data = {"support": support}

    diff = EdgeDifference(support_name="support")(inputs, field_data=field_data)
    actual = EdgeToNodeSum(support_name="support")(
        phlower_tensor_collection({"edge": diff}), field_data=field_data
    )

    degrees = adjacency.sum(axis=1, keepdims=True)
    desired = adjacency @ h - degrees * h
    np.testing.assert_array_almost_equal(actual.to_numpy(), desired, decimal=5)


def test__sum_of_gathers_matches_dense_formula():
    n_nodes = 10
    n_feature = 3
    support = _random_support(n_nodes)
    adjacency = _adjacency_without_self_loops(support)
    h = np.random.rand(n_nodes, n_feature).astype(np.float32)
    inputs = phlower_tensor_collection(
        {"h": phlower_tensor(torch.from_numpy(h))}
    )
    field_data = {"support": support}

    gathered = EdgeGather(support_name="support")(inputs, field_data=field_data)
    actual = EdgeToNodeSum(support_name="support")(
        phlower_tensor_collection({"edge": gathered}),
        field_data=field_data,
    )

    degrees = adjacency.sum(axis=1, keepdims=True)
    desired = np.concatenate([degrees * h, adjacency @ h], axis=-1)
    np.testing.assert_array_almost_equal(actual.to_numpy(), desired, decimal=5)


def test__keeps_physical_dimension():
    n_nodes = 10
    support = _random_support(n_nodes)
    n_edges = len(support.values()) - n_nodes
    inputs = phlower_tensor_collection(
        {"edge": phlower_tensor(torch.rand(n_edges, 3), dimension={"L": 1})}
    )

    model = EdgeToNodeSum(support_name="support")
    actual = model(inputs, field_data={"support": support})

    assert actual.dimension.to_dict()["L"] == 1.0


def test__gradient_flows_to_input():
    n_nodes = 10
    support = _random_support(n_nodes)
    n_edges = len(support.values()) - n_nodes
    edge = torch.rand(n_edges, 3, requires_grad=True)
    inputs = phlower_tensor_collection({"edge": phlower_tensor(edge)})

    model = EdgeToNodeSum(support_name="support")
    actual = model(inputs, field_data={"support": support})
    actual.to_tensor().sum().backward()

    assert edge.grad is not None
    np.testing.assert_array_almost_equal(
        edge.grad.numpy(), np.ones((n_edges, 3))
    )


def test__raise_error_when_edge_count_mismatch():
    n_nodes = 10
    support = _random_support(n_nodes)
    n_edges = len(support.values()) - n_nodes
    inputs = phlower_tensor_collection(
        {"edge": phlower_tensor(torch.rand(n_edges + 1, 3))}
    )

    model = EdgeToNodeSum(support_name="support")
    with pytest.raises(ValueError) as ex:
        model(inputs, field_data={"support": support})

    assert "does not match" in str(ex.value)


def test__time_series_is_processed_stepwise():
    support = _path_graph_support()
    ts = torch.rand(4, 4, 2)  # 4 steps, 4 edges
    inputs = phlower_tensor_collection(
        {"edge": phlower_tensor(ts, is_time_series=True)}
    )

    model = EdgeToNodeSum(support_name="support")
    actual = model(inputs, field_data={"support": support})

    assert actual.is_time_series
    for t in range(ts.shape[0]):
        step = phlower_tensor_collection({"edge": phlower_tensor(ts[t])})
        desired = model(step, field_data={"support": support})
        np.testing.assert_array_almost_equal(
            actual.to_numpy()[t], desired.to_numpy()
        )


def test__raise_error_when_voxel():
    support = _path_graph_support()
    inputs = phlower_tensor_collection(
        {"edge": phlower_tensor(torch.rand(2, 2, 2, 3), is_voxel=True)}
    )

    model = EdgeToNodeSum(support_name="support")
    with pytest.raises(ValueError) as ex:
        model(inputs, field_data={"support": support})

    assert "does not support voxel" in str(ex.value)
