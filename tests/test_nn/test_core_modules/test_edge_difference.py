import numpy as np
import pytest
import torch
from phlower_tensor import PhlowerTensor, phlower_tensor
from phlower_tensor.collections import phlower_tensor_collection
from phlower_tensor.functionals import to_batch

from phlower.nn import EdgeDifference
from phlower.settings._module_settings import EdgeDifferenceSetting


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
    model = EdgeDifference(support_name="support")

    # To check EdgeDifference inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize("support_name", ["aaa", "support1"])
@pytest.mark.parametrize("with_norm", [True, False])
def test__can_pass_parameters_via_setting(support_name: str, with_norm: bool):
    setting = EdgeDifferenceSetting(
        support_name=support_name, with_norm=with_norm
    )
    model = EdgeDifference.from_setting(setting)

    assert model._support_name == support_name
    assert model._with_norm is with_norm


def test__difference_on_path_graph():
    support = _path_graph_support()
    h = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 5.0]], dtype=np.float32)
    inputs = phlower_tensor_collection(
        {"h": phlower_tensor(torch.from_numpy(h))}
    )

    model = EdgeDifference(support_name="support")
    actual = model(inputs, field_data={"support": support})

    # coalesced edge order: (0, 1), (1, 0), (1, 2), (2, 1)
    desired = np.array([h[1] - h[0], h[0] - h[1], h[2] - h[1], h[1] - h[2]])
    np.testing.assert_array_almost_equal(actual.to_numpy(), desired)


def test__batch_correct():
    # a batched support is block-diagonal, so samples must not interact
    supports = [_path_graph_support(), _random_support(5)]
    hs = [torch.rand(3, 2), torch.rand(5, 2)]
    batched_support, _ = to_batch(supports)
    batched_inputs = phlower_tensor_collection(
        {"h": phlower_tensor(torch.concatenate(hs))}
    )

    model = EdgeDifference(support_name="support")
    actual = model(batched_inputs, field_data={"support": batched_support})

    desired = np.concatenate(
        [
            model(
                phlower_tensor_collection({"h": phlower_tensor(h)}),
                field_data={"support": support},
            ).to_numpy()
            for support, h in zip(supports, hs, strict=True)
        ]
    )
    np.testing.assert_array_almost_equal(actual.to_numpy(), desired)


@pytest.mark.parametrize("n_feature", [1, 3, 8])
def test__with_norm_appends_l2_norm(n_feature: int):
    n_nodes = 10
    support = _random_support(n_nodes)
    inputs = phlower_tensor_collection(
        {"h": phlower_tensor(torch.rand(n_nodes, n_feature))}
    )
    field_data = {"support": support}

    plain = EdgeDifference(support_name="support")(
        inputs, field_data=field_data
    )
    actual = EdgeDifference(support_name="support", with_norm=True)(
        inputs, field_data=field_data
    )

    assert actual.shape == (plain.shape[0], n_feature + 1)
    np.testing.assert_array_almost_equal(
        actual.to_numpy()[:, :n_feature], plain.to_numpy()
    )
    np.testing.assert_array_almost_equal(
        actual.to_numpy()[:, -1],
        np.linalg.norm(plain.to_numpy(), axis=-1),
    )


def test__self_loops_are_ignored():
    n_nodes = 10
    with_loops = _random_support(n_nodes, with_self_loops=True)
    without_loops = phlower_tensor(
        torch.sparse_coo_tensor(
            with_loops.indices(),
            with_loops.values(),
            size=with_loops.shape,
        )
        .to_dense()
        .fill_diagonal_(0.0)
        .to_sparse()
    )
    inputs = phlower_tensor_collection(
        {"h": phlower_tensor(torch.rand(n_nodes, 3))}
    )

    model = EdgeDifference(support_name="support")
    actual = model(inputs, field_data={"support": with_loops})
    desired = model(inputs, field_data={"support": without_loops})

    np.testing.assert_array_almost_equal(actual.to_numpy(), desired.to_numpy())


def test__keeps_physical_dimension():
    n_nodes = 10
    support = _random_support(n_nodes)
    inputs = phlower_tensor_collection(
        {"h": phlower_tensor(torch.rand(n_nodes, 3), dimension={"L": 1})}
    )

    model = EdgeDifference(support_name="support", with_norm=True)
    actual = model(inputs, field_data={"support": support})

    assert actual.dimension.to_dict()["L"] == 1.0


def test__gradient_flows_to_input():
    n_nodes = 10
    support = _random_support(n_nodes)
    h = torch.rand(n_nodes, 3, requires_grad=True)
    inputs = phlower_tensor_collection({"h": phlower_tensor(h)})

    model = EdgeDifference(support_name="support", with_norm=True)
    actual = model(inputs, field_data={"support": support})
    actual.to_tensor().sum().backward()

    assert h.grad is not None
    assert torch.isfinite(h.grad).all()


def test__raise_error_when_with_norm_on_rank1_tensor():
    n_nodes = 10
    support = _random_support(n_nodes)
    inputs = phlower_tensor_collection(
        {"h": phlower_tensor(torch.rand(n_nodes, 3, 4))}
    )

    model = EdgeDifference(support_name="support", with_norm=True)
    with pytest.raises(ValueError) as ex:
        model(inputs, field_data={"support": support})

    assert "only applicable to a rank-0 tensor" in str(ex.value)


@pytest.mark.parametrize("with_norm", [False, True])
def test__time_series_is_processed_stepwise(with_norm: bool):
    support = _path_graph_support()
    ts = torch.rand(4, 3, 2)  # 4 steps, 3 nodes
    inputs = phlower_tensor_collection(
        {"h": phlower_tensor(ts, is_time_series=True)}
    )

    model = EdgeDifference(support_name="support", with_norm=with_norm)
    actual = model(inputs, field_data={"support": support})

    assert actual.is_time_series
    for t in range(ts.shape[0]):
        step = phlower_tensor_collection({"h": phlower_tensor(ts[t])})
        desired = model(step, field_data={"support": support})
        np.testing.assert_array_almost_equal(
            actual.to_numpy()[t], desired.to_numpy()
        )


def test__raise_error_when_voxel():
    support = _path_graph_support()
    inputs = phlower_tensor_collection(
        {"h": phlower_tensor(torch.rand(2, 2, 2, 3), is_voxel=True)}
    )

    model = EdgeDifference(support_name="support")
    with pytest.raises(ValueError) as ex:
        model(inputs, field_data={"support": support})

    assert "does not support voxel" in str(ex.value)
