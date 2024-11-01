import numpy as np
import pytest
from phlower import PhlowerTensor, phlower_tensor
from phlower._fields import SimulationField
from phlower.collections import phlower_tensor_collection
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn import IsoGCN


def generate_orthogonal_matrix() -> np.ndarray:
    def normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x)

    vec1 = normalize(np.random.rand(3) * 2 - 1)
    vec2 = normalize(np.random.rand(3) * 2 - 1)

    vec3 = normalize(np.cross(vec1, vec2))
    vec2 = np.cross(vec3, vec1)
    return np.array([vec1, vec2, vec3])


def generate_isoam_like_matrix(
    location: np.ndarray, key_names: list[str]
) -> IPhlowerTensorCollections:
    n_nodes = len(location)
    g = np.array(
        [
            [location[col] - location[row] for col in range(n_nodes)]
            for row in range(n_nodes)
        ]
    )
    g_eye = np.einsum("ij,ik->ijk", np.eye(n_nodes), np.sum(g, axis=1))
    g_tilde: np.ndarray = g - g_eye

    dict_data: dict[str, PhlowerTensor] = {}

    # print(f"{g_tilde.shape=}")
    n_axis = g_tilde.shape[-1]
    assert n_axis <= 3
    assert n_axis == len(key_names)

    for i in range(n_axis):
        dict_data[key_names[i]] = phlower_tensor(g_tilde[..., i])
    return phlower_tensor_collection(dict_data)


class OrthogonalOperator:
    def __init__(self):
        self._orthogonal = generate_orthogonal_matrix()

    def transform_position(self, location: np.ndarray) -> np.ndarray:
        return np.einsum("ij,nj->ni", self._orthogonal, location)

    def transform_rank1(self, array: np.ndarray) -> np.ndarray:
        return np.einsum("ip,npk->nik", self._orthogonal, array)

    def transform_rank2(self, array: np.ndarray) -> np.ndarray:
        # U X U^T
        _tmp = np.einsum("ij,njkl->nikl", self._orthogonal, array)
        return np.einsum("njkl,ki->njil", _tmp, self._orthogonal.T)


def forward_isogcn_with_no_weight(
    location: np.ndarray, propagations: list[str], feature: np.ndarray
) -> PhlowerTensor:
    support_names = ["isoam_x", "isoam_y", "isoam_z"]
    field = SimulationField(
        field_tensors=generate_isoam_like_matrix(location, support_names)
    )
    isogcn = IsoGCN(
        nodes=[],  # This is dummy because no weight is necessary
        isoam_names=support_names,
        propagations=propagations,
        use_self_network=False,
    )
    return isogcn.forward(
        phlower_tensor_collection({"h": phlower_tensor(feature)}),
        field_data=field,
    )


# region test for equivariance


@pytest.mark.parametrize("n_nodes, n_feature", [(4, 2), (5, 3), (7, 1)])
def test__equivariance_of_convolution_rank0_to_rank1(
    n_nodes: int, n_feature: int
):
    # common
    propagations = ["convolution"]

    # base coordination
    location = np.random.rand(n_nodes, 3)
    h = np.random.rand(n_nodes, n_feature)
    h_res = forward_isogcn_with_no_weight(
        location=location, propagations=propagations, feature=h
    )
    assert h_res.rank() == 1

    # for location applied by orthogonal matrix
    ortho_op = OrthogonalOperator()
    ortho_location = ortho_op.transform_position(location)
    ortho_h = h
    ortho_h_res = forward_isogcn_with_no_weight(
        location=ortho_location, propagations=propagations, feature=ortho_h
    )
    assert ortho_h_res.rank() == 1

    # Compare
    rotate_h = ortho_op.transform_rank1(h_res.numpy())
    np.testing.assert_array_almost_equal(rotate_h, ortho_h_res.numpy())


@pytest.mark.parametrize("n_nodes, n_feature", [(10, 3), (5, 1), (7, 3)])
def test__equivariance_of_convolution_rank0_to_rank2(
    n_nodes: int, n_feature: int
):
    # common
    propagations = ["convolution", "tensor_product"]

    # base coordination
    location = np.random.rand(n_nodes, 3)
    h = np.random.rand(n_nodes, n_feature)
    h_res = forward_isogcn_with_no_weight(
        location=location,
        propagations=propagations,
        feature=h,
    )
    assert h_res.rank() == 2

    # for location applied by orthogonal matrix
    ortho_op = OrthogonalOperator()
    ortho_location = ortho_op.transform_position(location)
    ortho_h = h
    ortho_h_res = forward_isogcn_with_no_weight(
        location=ortho_location, propagations=propagations, feature=ortho_h
    )
    assert ortho_h_res.rank() == 2

    # Compare
    # orthogonal transformation
    # U X U^T
    rotate_h = ortho_op.transform_rank2(h_res.numpy())
    np.testing.assert_array_almost_equal(rotate_h, ortho_h_res.numpy())


@pytest.mark.parametrize("n_nodes, n_feature", [(10, 3), (5, 1), (7, 3)])
def test__equivariance_of_laplacian_rank1_to_rank1(
    n_nodes: int, n_feature: int
):
    # common
    propagations = ["tensor_product", "contraction"]

    # base coordination
    location = np.random.rand(n_nodes, 3)
    h = np.random.rand(n_nodes, 3, n_feature)
    h_res = forward_isogcn_with_no_weight(
        location=location,
        propagations=propagations,
        feature=h,
    )
    assert h_res.rank() == 1

    # for location applied by orthogonal matrix
    ortho_op = OrthogonalOperator()
    ortho_location = ortho_op.transform_position(location)
    ortho_h = ortho_op.transform_rank1(h)
    ortho_h_res = forward_isogcn_with_no_weight(
        location=ortho_location, propagations=propagations, feature=ortho_h
    )
    assert ortho_h_res.rank() == 1

    # Compare
    rotate_h = ortho_op.transform_rank1(h_res.numpy())
    np.testing.assert_array_almost_equal(rotate_h, ortho_h_res.numpy())


@pytest.mark.parametrize("n_nodes, n_feature", [(10, 3), (5, 1), (7, 3)])
def test__invariance_of_contraction_rank1_to_rank0(
    n_nodes: int, n_feature: int
):
    # common
    propagations = ["contraction"]

    # base coordination
    location = np.random.rand(n_nodes, 3)
    h = np.random.rand(n_nodes, 3, n_feature)
    h_res = forward_isogcn_with_no_weight(
        location=location,
        propagations=propagations,
        feature=h,
    )
    assert h_res.rank() == 0

    # for location applied by orthogonal matrix
    ortho_op = OrthogonalOperator()
    ortho_location = ortho_op.transform_position(location)
    ortho_h = ortho_op.transform_rank1(h)
    ortho_h_res = forward_isogcn_with_no_weight(
        location=ortho_location, propagations=propagations, feature=ortho_h
    )
    assert ortho_h_res.rank() == 0

    # Compare
    np.testing.assert_array_almost_equal(h_res.numpy(), ortho_h_res.numpy())


@pytest.mark.parametrize("n_nodes, n_feature", [(10, 3), (5, 1), (7, 3)])
def test__equivariance_of_contraction_rank2_to_rank1(
    n_nodes: int, n_feature: int
):
    # common
    propagations = ["contraction"]

    # base coordination
    location = np.random.rand(n_nodes, 3)
    h = np.random.rand(n_nodes, 3, 3, n_feature)
    h_res = forward_isogcn_with_no_weight(
        location=location,
        propagations=propagations,
        feature=h,
    )
    assert h_res.rank() == 1

    # for location applied by orthogonal matrix
    ortho_op = OrthogonalOperator()
    ortho_location = ortho_op.transform_position(location)
    ortho_h = ortho_op.transform_rank2(h)
    ortho_h_res = forward_isogcn_with_no_weight(
        location=ortho_location, propagations=propagations, feature=ortho_h
    )
    assert ortho_h_res.rank() == 1

    # Compare
    rotate_h = ortho_op.transform_rank1(h_res.numpy())
    np.testing.assert_array_almost_equal(rotate_h, ortho_h_res.numpy())


@pytest.mark.parametrize("n_nodes, n_feature", [(10, 3), (5, 1), (7, 3)])
def test__invariance_of_contraction_rank2_to_rank0(
    n_nodes: int, n_feature: int
):
    # common
    propagations = ["contraction", "contraction"]

    # base coordination
    location = np.random.rand(n_nodes, 3)
    h = np.random.rand(n_nodes, 3, 3, n_feature)
    h_res = forward_isogcn_with_no_weight(
        location=location,
        propagations=propagations,
        feature=h,
    )
    assert h_res.rank() == 0

    # for location applied by orthogonal matrix
    ortho_op = OrthogonalOperator()
    ortho_location = ortho_op.transform_position(location)
    ortho_h = ortho_op.transform_rank2(h)
    ortho_h_res = forward_isogcn_with_no_weight(
        location=ortho_location, propagations=propagations, feature=ortho_h
    )
    assert ortho_h_res.rank() == 0

    # Compare
    np.testing.assert_array_almost_equal(h_res.numpy(), ortho_h_res.numpy())


# endregion
