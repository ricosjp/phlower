import numpy as np
import pytest
import torch
from phlower_tensor import (
    GraphBatchInfo,
    PhlowerTensor,
    SimulationField,
    phlower_tensor,
)
from phlower_tensor.collections import phlower_tensor_collection
from phlower_tensor.functionals import to_batch, unbatch
from scipy.spatial import KDTree

from phlower.nn._core_preset_group_modules import InterpolatorPresetGroupModule


def test__can_call_parameters():
    model = InterpolatorPresetGroupModule(
        source_position_name="",
        target_position_name="",
    )

    # To check the module inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "position_dimension",
    [
        None,
        {"L": 1},
    ],
)
@pytest.mark.parametrize("input_weight", [True, False])
@pytest.mark.parametrize("no_grad", [True, False])
def test__compute_sparse(
    position_dimension: dict[str, int] | None,
    input_weight: bool,
    no_grad: bool,
):
    length_scale = 0.1
    eps = 1.0e-2

    np_position = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0 + eps],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0 + eps],
            [0.0, 1.2, 0.0],
            [0.0, 1.2, 0.0 + eps],
            [0.0, 0.0, 1.3],
            [0.0, 0.0, 1.3 + eps],
        ]
    )[..., None]
    np_deformed_position = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0 + eps],
            [0.0, 1.2, 0.0],
            [0.0, 1.2, 0.0 + eps],
            [0.0, 0.0, 1.3],
            [0.0, 0.0, 1.3 + eps],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0 + eps],
        ]
    )[..., None]

    if input_weight:
        t_weight = torch.rand(len(np_position)) + 0.5
        if position_dimension is None:
            weight = phlower_tensor(t_weight, dimension=None)
        else:
            weight = phlower_tensor(t_weight, dimension={})
    else:
        weight = None

    b = np.exp(-eps / length_scale)
    desired = np.array(
        [
            [1, b, 0, 0, 0, 0, 0, 0],
            [b, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, b, 0, 0],
            [0, 0, 0, 0, b, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, b],
            [0, 0, 0, 0, 0, 0, b, 1],
            [0, 0, 1, b, 0, 0, 0, 0],
            [0, 0, b, 1, 0, 0, 0, 0],
        ]
    )
    if input_weight:
        desired = np.einsum("ij,j->ij", desired, weight.numpy())

    position = phlower_tensor(
        torch.from_numpy(np_position),
        dimension=position_dimension,
    )
    deformed_position = phlower_tensor(
        torch.from_numpy(np_deformed_position),
        dimension=position_dimension,
    )

    model = InterpolatorPresetGroupModule(
        source_position_name="",
        target_position_name="",
        k_nns=2,
        recompute_distance_k_nns_if_grad_enabled=not no_grad,
        recompute_distance_d_nns_if_grad_enabled=not no_grad,
    )
    if no_grad:
        with torch.no_grad():
            actual, distances = model._compute_sparse(
                position,
                deformed_position,
                inv_length_scale=1 / length_scale,
                weight=weight,
            )
    else:
        actual, distances = model._compute_sparse(
            position,
            deformed_position,
            inv_length_scale=1 / length_scale,
            weight=weight,
        )

    np.testing.assert_almost_equal(
        actual.to_tensor().to_dense().numpy(),
        desired,
    )

    # Test distance computation is correct
    tree = KDTree(position.numpy()[..., 0])
    d, _ = tree.query(deformed_position[..., 0], k=2)
    np.testing.assert_almost_equal(distances.numpy(), d)


@pytest.mark.parametrize("no_grad", [True, False])
def test__compute_sparse_coarsened(no_grad: bool):
    length_scale = 2.0

    np_position = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 5.0, 5.0],
            [0.0, 5.0, 6.0],
            [0.0, 5.0, 7.0],
        ]
    )[..., None]
    np_deformed_position = np.array(
        [
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 5.0, 6.0],
            [0.0, 5.0, 7.0],
        ]
    )[..., None]

    d0p0 = 1.0
    d0p5 = np.exp(-0.5 / length_scale)
    d1p0 = np.exp(-1.0 / length_scale)
    d2p0 = np.exp(-2.0 / length_scale)
    desired = np.array(
        [
            [d0p5, d1p0, 0.0, 0.0],
            [d0p0, d0p5, 0.0, 0.0],
            [d0p5, d0p0, 0.0, 0.0],
            [0.0, 0.0, d1p0, d2p0],
            [0.0, 0.0, d0p0, d1p0],
            [0.0, 0.0, d1p0, d0p0],
        ]
    )

    position = phlower_tensor(
        torch.from_numpy(np_position),
    )
    deformed_position = phlower_tensor(
        torch.from_numpy(np_deformed_position),
    )

    model = InterpolatorPresetGroupModule(
        source_position_name="",
        target_position_name="",
        k_nns=2,
        recompute_distance_k_nns_if_grad_enabled=not no_grad,
        recompute_distance_d_nns_if_grad_enabled=not no_grad,
    )
    if no_grad:
        with torch.no_grad():
            actual, distances = model._compute_sparse(
                deformed_position,
                position,
                inv_length_scale=1 / length_scale,
            )
    else:
        actual, distances = model._compute_sparse(
            deformed_position,
            position,
            inv_length_scale=1 / length_scale,
        )
    np.testing.assert_almost_equal(
        actual.to_tensor().to_dense().numpy(),
        desired,
    )

    # Test distance computation is correct
    tree = KDTree(deformed_position.numpy()[..., 0])
    d, _ = tree.query(position[..., 0], k=2)
    np.testing.assert_almost_equal(distances.numpy(), d)


@pytest.mark.parametrize(
    "position_dimension",
    [
        None,
        {"L": 1},
    ],
)
@pytest.mark.parametrize("input_weight", [True, False])
@pytest.mark.parametrize("no_grad", [True, False])
def test__compute_sparse_distance(
    position_dimension: dict[str, int] | None, input_weight: bool, no_grad: bool
):
    length_scale = 0.1
    eps = 1.0e-2
    distance = 0.5

    np_position = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0 + eps],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0 + eps],
            [0.0, 1.2, 0.0],
            [0.0, 1.2, 0.0 + eps],
            [0.0, 0.0, 1.3],
            [0.0, 0.0, 1.3 + eps],
        ]
    )[..., None]
    np_deformed_position = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0 + eps],
            [0.0, 1.2, 0.0],
            [0.0, 1.2, 0.0 + eps],
            [0.0, 0.0, 1.3],
            [0.0, 0.0, 1.3 + eps],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0 + eps],
        ]
    )[..., None]

    if input_weight:
        t_weight = torch.rand(len(np_position)) + 0.5
        if position_dimension is None:
            weight = phlower_tensor(t_weight, dimension=None)
        else:
            weight = phlower_tensor(t_weight, dimension={})
    else:
        weight = None

    position = phlower_tensor(
        torch.from_numpy(np_position),
        dimension=position_dimension,
    )
    deformed_position = phlower_tensor(
        torch.from_numpy(np_deformed_position),
        dimension=position_dimension,
    )

    model = InterpolatorPresetGroupModule(
        source_position_name="",
        target_position_name="",
        d_nns=distance,
        recompute_distance_k_nns_if_grad_enabled=not no_grad,
        recompute_distance_d_nns_if_grad_enabled=not no_grad,
    )
    if no_grad:
        with torch.no_grad():
            actual, distances = model._compute_sparse_dist(
                position,
                deformed_position,
                inv_length_scale=1 / length_scale,
                weight=weight,
            )
    else:
        actual, distances = model._compute_sparse_dist(
            position,
            deformed_position,
            inv_length_scale=1 / length_scale,
            weight=weight,
        )

    # Test distance computation is correct
    original_tree = KDTree(position.numpy()[..., 0])
    deformed_tree = KDTree(deformed_position.numpy()[..., 0])
    desired_sp = original_tree.sparse_distance_matrix(
        deformed_tree,
        distance,
        output_type="coo_matrix",
    )
    desired_distances = desired_sp.data
    np.testing.assert_almost_equal(
        np.ravel(distances.numpy()),
        desired_distances,
    )

    # Test sparse matrix is correct
    desired_sp.data = np.exp(-desired_sp.data / length_scale)
    desired_sp = desired_sp.toarray()
    if input_weight:
        desired_sp = np.einsum("ij,j->ij", desired_sp, weight)
    np.testing.assert_almost_equal(
        actual.to_tensor().to_dense().numpy(),
        desired_sp,
    )


@pytest.mark.parametrize("no_grad", [True, False])
def test__compute_sparse_dist_coarsened(no_grad: bool):
    length_scale = 2.0

    np_position = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 5.0, 5.0],
            [0.0, 5.0, 6.0],
            [0.0, 5.0, 7.0],
        ]
    )[..., None]
    np_deformed_position = np.array(
        [
            [0.5, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 5.0, 6.0],
            [0.0, 5.0, 7.0],
        ]
    )[..., None]

    position = phlower_tensor(
        torch.from_numpy(np_position),
    )
    deformed_position = phlower_tensor(
        torch.from_numpy(np_deformed_position),
    )

    model = InterpolatorPresetGroupModule(
        source_position_name="",
        target_position_name="",
        conservative=False,
        bidirectional=False,
        k_nns=2,
        d_nns=2.1,
        recompute_distance_k_nns_if_grad_enabled=not no_grad,
        recompute_distance_d_nns_if_grad_enabled=not no_grad,
    )
    if no_grad:
        with torch.no_grad():
            actual, distances = model._compute_sparse_dist(
                position,
                deformed_position,
                inv_length_scale=1 / length_scale,
            )
    else:
        actual, distances = model._compute_sparse_dist(
            position,
            deformed_position,
            inv_length_scale=1 / length_scale,
        )

    # Test distance computation is correct
    original_tree = KDTree(position.numpy()[..., 0])
    deformed_tree = KDTree(deformed_position.numpy()[..., 0])
    desired_sp = original_tree.sparse_distance_matrix(
        deformed_tree,
        2.1,
        output_type="coo_matrix",
    )
    desired_distances = desired_sp.data
    np.testing.assert_almost_equal(
        np.ravel(distances.numpy()),
        desired_distances,
    )

    # Test sparse matrix is correct
    desired_sp.data = np.exp(-desired_sp.data / length_scale)
    desired_sp = desired_sp.toarray()
    np.testing.assert_almost_equal(
        actual.to_tensor().to_dense().numpy(),
        desired_sp,
    )


@pytest.mark.parametrize("has_weight", [True, False])
def test__compute_sparse_grad(has_weight: bool):
    position = phlower_tensor(torch.randn(10, 3, 1))
    deformed_position = phlower_tensor(torch.randn(10, 3, 1))
    if has_weight:
        weight = phlower_tensor(torch.randn(10, 1))
    else:
        weight = None

    model = InterpolatorPresetGroupModule(
        source_position_name="",
        target_position_name="",
        recompute_distance_k_nns_if_grad_enabled=True,
        recompute_distance_d_nns_if_grad_enabled=False,
    )

    inv_length_scale = torch.nn.Parameter(
        torch.tensor([1.0], dtype=torch.float32), requires_grad=True
    )
    actual, _ = model._compute_sparse(
        position,
        deformed_position,
        inv_length_scale=inv_length_scale,
        weight=weight,
    )
    loss = torch.sum(actual)
    loss.backward()
    assert inv_length_scale.grad is not None


@pytest.mark.parametrize("has_weight", [True, False])
def test__compute_sparse_dist_grad(has_weight: bool):
    position = phlower_tensor(torch.randn(10, 3, 1))
    deformed_position = phlower_tensor(torch.randn(10, 3, 1))
    if has_weight:
        weight = phlower_tensor(torch.randn(10, 1))
    else:
        weight = None

    model = InterpolatorPresetGroupModule(
        source_position_name="",
        target_position_name="",
        d_nns=0.1,
        recompute_distance_k_nns_if_grad_enabled=False,
        recompute_distance_d_nns_if_grad_enabled=True,
    )

    inv_length_scale = torch.nn.Parameter(
        torch.tensor([1.0], dtype=torch.float32), requires_grad=True
    )
    actual, _ = model._compute_sparse_dist(
        position,
        deformed_position,
        inv_length_scale=inv_length_scale,
        weight=weight,
    )
    loss = torch.sum(actual)
    loss.backward()
    assert inv_length_scale.grad is not None


@pytest.mark.parametrize("shape", [(4, 2), (4, 3, 2)])
@pytest.mark.parametrize("dimension", [None, {}, {"L": 1, "T": 1}])
@pytest.mark.parametrize("normalize_row", [True, False])
@pytest.mark.parametrize("conservative", [True, False])
def test__compute_normalized_spmm(
    shape: tuple[int],
    dimension: dict[str, int],
    normalize_row: bool,
    conservative: bool,
):
    model = InterpolatorPresetGroupModule(
        source_position_name="",
        target_position_name="",
        normalize_row=normalize_row,
        conservative=conservative,
        bidirectional=conservative,
    )

    np_sparse = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [0, 1, 1, 3],
            [1, 0, 2, 0],
        ]
    )
    if dimension is None:
        sparse_dimension = None
    else:
        sparse_dimension = {}
    sparse = phlower_tensor(
        torch.tensor(np_sparse).to_sparse_coo().to(dtype=torch.float32),
        dimension=sparse_dimension,
    )
    variable = phlower_tensor(torch.randn(shape), dimension=dimension)
    actual = model._compute_normalized_spmm(sparse, variable)

    if normalize_row:
        np_weight_row = 1 / np.sum(np_sparse, axis=1)
    else:
        np_weight_row = np.ones(sparse.shape[0])

    if conservative:
        np_weight_col = (1 / (np_sparse.T @ np_weight_row[:, None]))[..., 0]
    else:
        np_weight_col = np.ones(np_weight_row.shape)

    desired = np.einsum(
        "i,ij,j,j...->i...",
        np_weight_row,
        np_sparse,
        np_weight_col,
        variable.numpy(),
    )

    np.testing.assert_almost_equal(actual.numpy(), desired, decimal=6)

    if conservative:
        # Test conservation
        total_before = np.sum(actual.numpy(), axis=0)
        total_after = np.sum(desired, axis=0)
        np.testing.assert_almost_equal(total_after, total_before, decimal=6)

    # Check matrix consistensy to confirm the test script is correct
    matrix = np.einsum("i,ij,j->ij", np_weight_row, np_sparse, np_weight_col)
    if conservative:
        summed = np.sum(matrix, axis=0)
        np.testing.assert_almost_equal(summed, 1.0)
    else:
        if normalize_row:
            summed = np.sum(matrix, axis=1)
            np.testing.assert_almost_equal(summed, 1.0)


@pytest.mark.parametrize("unbatch", [True, False])
def test__interpolator_input_to_output_map(unbatch: bool):
    input_to_output_map = {"s_in": "s_out", "t_in": "t_out"}
    model = InterpolatorPresetGroupModule(
        source_position_name="deformed_position",
        target_position_name="position",
        source_field_name="position",
        variable_names=list(input_to_output_map.keys()),
        k_nns=2,
        unbatch=unbatch,
        input_to_output_map=input_to_output_map,
    )

    # NOTE: Double the position to get the identity map with k_nns = 2
    rand = torch.randn(4, 3, 1)
    position = phlower_tensor(torch.cat([rand, rand], dim=0))

    ts = phlower_tensor_collection(
        {"s_in": position * 3, "t_in": torch.linalg.norm(position, dim=1)}
    )
    batch_info = GraphBatchInfo(
        sizes=[position.size],
        shapes=[position.shape],
        n_nodes=[position.n_vertices()],
        dense_concat_dim=0,
    )
    field = SimulationField(
        phlower_tensor_collection(
            {"position": position, "deformed_position": position}
        ),
        batch_info={"position": batch_info, "deformed_position": batch_info},
    )
    ret = model(ts, field_data=field)

    assert set(ret.keys()) == set(input_to_output_map.values())

    for input_name, output_name in input_to_output_map.items():
        desired = ts[input_name]
        actual = ret[output_name]
        np.testing.assert_almost_equal(actual.numpy(), desired.numpy())


@pytest.mark.parametrize("shape", [(4, 16), (4, 3, 16)])
@pytest.mark.parametrize("k_nns", [2, 4])
@pytest.mark.parametrize("dimension", [None, {}, {"L": 1, "T": 1}])
@pytest.mark.parametrize(
    "conservative, bidirectional",
    [
        (False, True),
        (False, False),
        # If conservative, should be bidirectional
        (True, True),
    ],
)
@pytest.mark.parametrize("n_head", [1, 4, 16])
@pytest.mark.parametrize("weight_name", [None, "length"])
@pytest.mark.parametrize("d_nns", [None, 0.1])
@pytest.mark.parametrize("no_grad", [True, False])
def test__interpolator_dimension(
    shape: tuple[int],
    k_nns: int,
    dimension: dict[str, int] | None,
    conservative: bool,
    bidirectional: bool,
    n_head: int,
    weight_name: str | None,
    d_nns: float | None,
    no_grad: bool,
):
    model = InterpolatorPresetGroupModule(
        source_position_name="deformed_position",
        target_position_name="position",
        source_field_name="position",
        conservative=conservative,
        bidirectional=bidirectional,
        source_position_from_field=False,
        k_nns=k_nns,
        d_nns=d_nns,
        n_head=n_head,
        weight_name=weight_name,
        recompute_distance_k_nns_if_grad_enabled=not no_grad,
        recompute_distance_d_nns_if_grad_enabled=not no_grad,
    )

    if dimension is None:
        length_dimension = None
    else:
        length_dimension = {"L": 1}
    position = phlower_tensor(torch.randn(4, 3, 1), dimension=length_dimension)
    deformation = phlower_tensor(
        torch.randn(4, 3, 16),
        dimension=length_dimension,
    )
    length = phlower_tensor(
        torch.rand((4, 1)) + 0.5, dimension=length_dimension
    )

    variable = phlower_tensor(torch.rand(shape), dimension=dimension)
    ts = phlower_tensor_collection(
        {"t": variable, "deformed_position": position + deformation}
    )
    batch_info = GraphBatchInfo(
        sizes=[position.size],
        shapes=[position.shape],
        n_nodes=[position.n_vertices()],
        dense_concat_dim=0,
    )
    field = SimulationField(
        phlower_tensor_collection({"position": position, "length": length}),
        batch_info={"position": batch_info},
    )

    if no_grad:
        with torch.no_grad():
            ret = model(ts, field_data=field)["t"]
    else:
        ret = model(ts, field_data=field)["t"]

    assert ret.dimension == variable.dimension


@pytest.mark.parametrize("shape", [(4, 16), (4, 3, 16)])
@pytest.mark.parametrize("k_nns", [2, 4])
@pytest.mark.parametrize("dimension", [None, {}, {"L": 1, "T": 1}])
@pytest.mark.parametrize(
    "conservative, bidirectional",
    [
        (False, True),
        (False, False),
        # If conservative, should be bidirectional
        (True, True),
    ],
)
@pytest.mark.parametrize("n_head", [1, 4, 16])
@pytest.mark.parametrize("weight_name", [None, "length"])
@pytest.mark.parametrize("d_nns", [None, 0.1])
def test__interpolator_grad(
    shape: tuple[int],
    k_nns: int,
    dimension: dict[str, int] | None,
    conservative: bool,
    bidirectional: bool,
    n_head: int,
    weight_name: str | None,
    d_nns: float | None,
):
    model = InterpolatorPresetGroupModule(
        source_position_name="deformed_position",
        target_position_name="position",
        source_field_name="position",
        learnable_length_scale=True,
        conservative=conservative,
        bidirectional=bidirectional,
        source_position_from_field=False,
        k_nns=k_nns,
        d_nns=d_nns,
        n_head=n_head,
        weight_name=weight_name,
        recompute_distance_k_nns_if_grad_enabled=True,
        recompute_distance_d_nns_if_grad_enabled=True,
    )

    if dimension is None:
        length_dimension = None
    else:
        length_dimension = {"L": 1}
    position = phlower_tensor(torch.randn(4, 3, 1), dimension=length_dimension)
    deformation = phlower_tensor(
        torch.randn(4, 3, 16),
        dimension=length_dimension,
    )
    length = phlower_tensor(
        torch.rand((4, 1)) + 0.5, dimension=length_dimension
    )

    variable = phlower_tensor(torch.rand(shape), dimension=dimension)
    ts = phlower_tensor_collection(
        {"t": variable, "deformed_position": position + deformation}
    )
    batch_info = GraphBatchInfo(
        sizes=[position.size],
        shapes=[position.shape],
        n_nodes=[position.n_vertices()],
        dense_concat_dim=0,
    )
    field = SimulationField(
        phlower_tensor_collection({"position": position, "length": length}),
        batch_info={"position": batch_info},
    )

    ret = model(ts, field_data=field)["t"]
    loss = torch.sum(ret)
    loss.backward()
    assert model._inv_length_scale.grad is not None


@pytest.mark.parametrize("shape", [(4, 16), (4, 3, 16)])
@pytest.mark.parametrize("dimension", [None, {}, {"L": 1, "T": 1}])
@pytest.mark.parametrize("n_head", [1, 4, 16])
@pytest.mark.parametrize("deformation_scale", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("weight_name", [None, "length"])
@pytest.mark.parametrize("d_nns", [None, 0.1])
@pytest.mark.parametrize("no_grad", [True, False])
def test__interpolator_conservation(
    shape: tuple[int],
    dimension: dict[str, int] | None,
    n_head: int,
    deformation_scale: float,
    weight_name: str | None,
    d_nns: float | None,
    no_grad: bool,
):
    model = InterpolatorPresetGroupModule(
        source_position_name="deformed_position",
        target_position_name="position",
        source_field_name="position",
        conservative=True,
        bidirectional=True,
        source_position_from_field=False,
        k_nns=2,
        d_nns=d_nns,
        n_head=n_head,
        weight_name=weight_name,
        recompute_distance_k_nns_if_grad_enabled=not no_grad,
        recompute_distance_d_nns_if_grad_enabled=not no_grad,
    )

    if dimension is None:
        length_dimension = None
    else:
        length_dimension = {"L": 1}
    position = phlower_tensor(torch.randn(4, 3, 1), dimension=length_dimension)
    deformation = (
        phlower_tensor(
            torch.randn(4, 3, 16),
            dimension=length_dimension,
        )
        * deformation_scale
    )
    length = phlower_tensor(
        torch.rand((4, 1)) + 0.5, dimension=length_dimension
    )

    variable = phlower_tensor(torch.rand(shape), dimension=dimension)
    desired_total = torch.sum(variable, dim=0)
    ts = phlower_tensor_collection(
        {"t": variable, "deformed_position": position + deformation}
    )
    batch_info = GraphBatchInfo(
        sizes=[position.size],
        shapes=[position.shape],
        n_nodes=[position.n_vertices()],
        dense_concat_dim=0,
    )
    field = SimulationField(
        phlower_tensor_collection({"position": position, "length": length}),
        batch_info={"position": batch_info},
    )

    if no_grad:
        with torch.no_grad():
            ret = model(ts, field_data=field)["t"]
    else:
        ret = model(ts, field_data=field)["t"]

    actual_total = torch.sum(ret, dim=0)
    np.testing.assert_almost_equal(
        actual_total.numpy(),
        desired_total.numpy(),
        decimal=2,
    )


@pytest.mark.parametrize("feature_shape", [[16], [3, 16]])
@pytest.mark.parametrize("n_head", [1, 4, 16])
@pytest.mark.parametrize("n_batch", [1, 2, 3, 6])
@pytest.mark.parametrize("is_time_series", [True, False])
@pytest.mark.parametrize("conservative", [True, False])
@pytest.mark.parametrize("dimension", [None, {}, {"L": 1, "T": 1}])
def test__batch_correct(
    feature_shape: list[int],
    n_head: int,
    n_batch: int,
    is_time_series: bool,
    conservative: bool,
    dimension: dict[str, int] | None,
):
    model = InterpolatorPresetGroupModule(
        source_position_name="deformed_position",
        target_position_name="position",
        source_field_name="position",
        conservative=conservative,
        bidirectional=conservative,
        source_position_from_field=False,
        k_nns=2,
        n_head=n_head,
    )

    n_points = np.random.randint(10, 20, n_batch)
    time_series_length = 5

    if dimension is None:
        length_dimension = None
    else:
        length_dimension = {"L": 1}
    positions = [
        phlower_tensor(torch.randn(n, 3, 1), dimension=length_dimension)
        for n in n_points
    ]
    deformations = [
        phlower_tensor(torch.randn(n, 3, 1), length_dimension) for n in n_points
    ]
    deformed_positions = [
        p + d for p, d in zip(positions, deformations, strict=True)
    ]
    if is_time_series:
        variables = [
            phlower_tensor(
                torch.rand([time_series_length, n] + feature_shape),
                dimension=dimension,
                is_time_series=True,
            )
            for n in n_points
        ]
        batched_variable, _ = to_batch(variables, dense_concat_dim=1)
    else:
        variables = [
            phlower_tensor(torch.rand([n] + feature_shape), dimension=dimension)
            for n in n_points
        ]
        batched_variable, _ = to_batch(variables, dense_concat_dim=0)

    batched_deformed_position, _ = to_batch(deformed_positions)
    batched_position, batch_info = to_batch(positions)
    ts = phlower_tensor_collection(
        {"t": batched_variable, "deformed_position": batched_deformed_position}
    )
    field = SimulationField(
        phlower_tensor_collection({"position": batched_position}),
        batch_info={"position": batch_info},
    )
    batched_prediction = model(ts, field_data=field)
    unbatched_predictions = unbatch(
        batched_prediction, batch_info=batch_info, n_nodes=n_points
    )
    for position, deformed_position, variable, prediction in zip(
        positions,
        deformed_positions,
        variables,
        unbatched_predictions,
        strict=True,
    ):
        ts = phlower_tensor_collection(
            {
                "t": variable,
                "deformed_position": deformed_position,
            }
        )
        _, batch_info = to_batch([position])
        field = SimulationField(
            phlower_tensor_collection({"position": position}),
            batch_info={"position": batch_info},
        )
        desired_prediction = model(ts, field_data=field)
        np.testing.assert_almost_equal(
            prediction["t"].numpy(), desired_prediction["t"].numpy(), decimal=5
        )


@pytest.mark.parametrize("k_nns", [5, 10])
@pytest.mark.parametrize("length_scale", [0.1, 0.2])
@pytest.mark.parametrize("n_source_points", [100, 150])
@pytest.mark.parametrize("n_target_points", [100, 120])
@pytest.mark.parametrize(
    "conservative, bidirectional, smaller_target",
    [
        (False, False, False),
        (False, True, False),
        # If smaller target domain, bidirectional is not recommended
        (False, False, True),
        # If conservative, should be bidirectional and equal domain
        (True, True, False),
    ],
)
# @pytest.mark.parametrize("bidirectional", [True])
@pytest.mark.parametrize("no_grad", [True, False])
@pytest.mark.parametrize("d_nns", [None, 0.1])
def test__interpolation_correct(
    k_nns: int,
    length_scale: float,
    n_source_points: int,
    n_target_points: int,
    conservative: bool,
    bidirectional: bool,
    smaller_target: bool,
    no_grad: bool,
    d_nns: float | None,
):
    x_source = phlower_tensor(
        torch.linspace(0.0, 1.0, n_source_points, dtype=torch.float32)
    )
    point_source = phlower_tensor(
        torch.zeros((n_source_points, 3, 1), dtype=torch.float32)
    )
    point_source[:, 0, 0] = x_source
    if smaller_target:
        x_target = phlower_tensor(
            torch.rand(n_target_points, dtype=torch.float32) * 0.8 + 0.1
        )
    else:
        x_target = phlower_tensor(
            torch.linspace(0.0, 1.0, n_target_points, dtype=torch.float32)
        )
    point_target = phlower_tensor(
        torch.zeros((n_target_points, 3, 1), dtype=torch.float32)
    )
    point_target[:, 0, 0] = x_target
    filter_x = (0.1 < x_target.numpy()) | (x_target.numpy() < 0.9)

    model = InterpolatorPresetGroupModule(
        source_position_name="point_source",
        target_position_name="point_target",
        conservative=conservative,
        bidirectional=bidirectional,
        length_scale=length_scale,
        normalize_row=True,
        k_nns=k_nns,
        d_nns=d_nns,
        recompute_distance_k_nns_if_grad_enabled=not no_grad,
        recompute_distance_d_nns_if_grad_enabled=not no_grad,
    )

    def scalar_field(pos: PhlowerTensor) -> PhlowerTensor:
        x = pos[:, 0]
        return x * (x - 1)

    def vector_field(pos: PhlowerTensor) -> PhlowerTensor:
        x = pos[:, 0]
        return torch.stack(
            [
                torch.cos(x * torch.pi),
                x**2,
                torch.sin(x * torch.pi),
            ],
            axis=1,
        )

    def ts_scalar_field(pos: PhlowerTensor) -> PhlowerTensor:
        x = pos[:, 0]
        ts = torch.linspace(0.0, 1.0, 10)
        return phlower_tensor(
            torch.stack(
                [torch.sin(x * torch.pi) * torch.cos(t * torch.pi) for t in ts],
                dim=0,
            ),
            is_time_series=True,
        )

    scalar = scalar_field(point_source)
    vector = vector_field(point_source)
    ts = ts_scalar_field(point_source)
    tc = phlower_tensor_collection(
        {"scalar": scalar, "vector": vector, "ts": ts}
    )
    batch_info_source = GraphBatchInfo(
        sizes=[point_source.size],
        shapes=[point_source.shape],
        n_nodes=[point_source.n_vertices()],
        dense_concat_dim=0,
    )
    batch_info_target = GraphBatchInfo(
        sizes=[point_target.size],
        shapes=[point_target.shape],
        n_nodes=[point_target.n_vertices()],
        dense_concat_dim=0,
    )
    field = SimulationField(
        phlower_tensor_collection(
            {"point_source": point_source, "point_target": point_target}
        ),
        batch_info={
            "point_source": batch_info_source,
            "point_target": batch_info_target,
        },
    )

    if no_grad:
        with torch.no_grad():
            ret = model(tc, field_data=field)
    else:
        ret = model(tc, field_data=field)

    desired_scalar = scalar_field(point_target)
    desired_vector = vector_field(point_target)
    desired_ts = ts_scalar_field(point_target)
    desired_tc = phlower_tensor_collection(
        {"scalar": desired_scalar, "vector": desired_vector, "ts": desired_ts}
    )

    for k in ret.keys():
        if ret[k].is_time_series:
            filtered_ret = ret[k][:, filter_x]
            filtered_desired_tc = desired_tc[k][:, filter_x]
        else:
            filtered_ret = ret[k][filter_x]
            filtered_desired_tc = desired_tc[k][filter_x]

        relative_rmse = torch.sqrt(
            torch.mean((filtered_ret - filtered_desired_tc) ** 2)
        ) / torch.sqrt(torch.mean(filtered_desired_tc**2))
        assert relative_rmse < 0.05
