from typing import Any
from unittest import mock

import numpy as np
import pytest
from phlower import PhlowerTensor, phlower_tensor
from phlower._fields import SimulationField
from phlower.collections import phlower_tensor_collection
from phlower.nn import IsoGCN
from phlower.settings._module_settings import IsoGCNSetting

# region Tools for testing


def generate_orthogonal_matrix() -> np.ndarray:
    def normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x)

    vec1 = normalize(np.random.rand(3) * 2 - 1)
    vec2 = normalize(np.random.rand(3) * 2 - 1)

    vec3 = normalize(np.cross(vec1, vec2))
    vec2 = np.cross(vec3, vec1)
    return np.array([vec1, vec2, vec3], dtype=np.float32)


def generate_isoam_like_matrix(
    location: np.ndarray, key_names: list[str]
) -> dict[str, PhlowerTensor]:
    n_nodes = len(location)
    g = np.array(
        [
            [location[col] - location[row] for col in range(n_nodes)]
            for row in range(n_nodes)
        ],
        dtype=np.float32,
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

    return dict_data


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
    location: np.ndarray,
    propagations: list[str],
    feature: np.ndarray,
) -> PhlowerTensor:
    support_names = ["isoam_x", "isoam_y", "isoam_z"]
    field = SimulationField(
        field_tensors=phlower_tensor_collection(
            generate_isoam_like_matrix(location, support_names)
        )
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


# endregion

# region Test for parsing setting


@pytest.mark.parametrize(
    "content, desired_items",
    [
        (
            {
                "isoam_names": ["isoam1"],
                "propagations": ["convolution"],
                "mul_order": "a_hw",
                "to_symmetric": True,
            },
            {
                "nodes": None,
                "isoam_names": ["isoam1"],
                "propagations": ["convolution"],
                "mul_order": "a_hw",
                "to_symmetric": True,
                "use_self_network": False,
                "use_coefficient": False,
                "use_neumann": False,
            },
        ),
        (
            {
                "nodes": [10, 10, 10],
                "isoam_names": ["isoam1"],
                "propagations": ["contraction"],
                "mul_order": "ah_w",
                "to_symmetric": False,
                "self_network": {"activations": ["identity"], "bias": False},
                "coefficient_network": {
                    "activations": ["tanh", "tanh"],
                    "bias": True,
                },
            },
            {
                "nodes": [10, 10, 10],
                "isoam_names": ["isoam1"],
                "propagations": ["contraction"],
                "mul_order": "ah_w",
                "to_symmetric": False,
                "use_self_network": True,
                "self_network_activations": ["identity"],
                "self_network_dropouts": [],
                "self_network_bias": False,
                "use_coefficient": True,
                "coefficient_activations": ["tanh", "tanh"],
                "coefficient_dropouts": [],
                "coefficient_bias": True,
                "use_neumann": False,
            },
        ),
        (
            {
                "nodes": [10, 30],
                "isoam_names": ["isoam1"],
                "propagations": ["contraction"],
                "mul_order": "ah_w",
                "to_symmetric": False,
                "self_network": {
                    "activations": ["tanh"],
                    "dropouts": [0.1],
                    "bias": False,
                },
                "neumann_setting": {
                    "factor": 0.2,
                    "neumann_input_name": "neumann_value",
                    "inversed_moment_name": "inv1",
                },
            },
            {
                "nodes": [10, 30],
                "isoam_names": ["isoam1"],
                "propagations": ["contraction"],
                "mul_order": "ah_w",
                "to_symmetric": False,
                "use_self_network": True,
                "self_network_activations": ["tanh"],
                "self_network_dropouts": [0.1],
                "self_network_bias": False,
                "use_coefficient": False,
                "use_neumann": True,
                "neumann_factor": 0.2,
                "neumann_input_name": "neumann_value",
                "inversed_moment_name": "inv1",
            },
        ),
    ],
)
def test__passed_correctly_from_setting(
    content: dict, desired_items: dict[str, Any]
):
    setting = IsoGCNSetting(**content)

    with mock.patch.object(IsoGCN, "__init__") as mocked:
        mocked.side_effect = lambda **x: None
        _ = IsoGCN.from_setting(setting)

        mocked.assert_called_once()

        kwards: dict = mocked.call_args.kwargs
        for k, v in desired_items.items():
            assert kwards.get(k) == v


# endregion

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
    np.testing.assert_array_almost_equal(
        rotate_h, ortho_h_res.numpy(), decimal=5
    )


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
    np.testing.assert_array_almost_equal(
        rotate_h, ortho_h_res.numpy(), decimal=5
    )


@pytest.mark.parametrize("n_nodes, n_feature", [(10, 3), (5, 1), (7, 3)])
def test__equivariance_of_rotation_rank1_to_rank1(n_nodes: int, n_feature: int):
    # common
    propagations = ["rotation"]

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
    np.testing.assert_array_almost_equal(
        rotate_h, ortho_h_res.numpy(), decimal=5
    )


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
    np.testing.assert_array_almost_equal(
        h_res.numpy(), ortho_h_res.numpy(), decimal=5
    )


# endregion


# region test for fobidding nonlinear operation
#  for tensor of which rank is greater than and equal 1


@pytest.mark.parametrize(
    "activations, dropouts, bias",
    [(["tanh"], [], False), ([], [], True), ([], [0.1], False)],
)
@pytest.mark.parametrize(
    "n_nodes, feature_dims, propagations",
    [
        (10, (10, 3, 5), ["convolution"]),
        (10, (10, 3, 5), ["contraction"]),
        (5, (5, 3, 3, 4), ["contraction"]),
        (10, (10, 3, 3, 3, 4), ["contraction", "contraction"]),
    ],
)
def test__forbid_operation_for_tensor_with_a_hw(
    activations: list[str],
    dropouts: list[float],
    bias: bool,
    n_nodes: int,
    feature_dims: tuple[int],
    propagations: list[str],
):
    location = np.random.rand(n_nodes, 3)
    h = np.random.rand(*feature_dims)

    support_names = ["isoam_x", "isoam_y", "isoam_z"]
    field = SimulationField(
        field_tensors=phlower_tensor_collection(
            generate_isoam_like_matrix(location, support_names)
        )
    )
    isogcn = IsoGCN(
        nodes=[feature_dims[-1], 10],
        mul_order="a_hw",
        isoam_names=support_names,
        propagations=propagations,
        use_self_network=True,
        self_network_activations=activations,
        self_network_dropouts=dropouts,
        self_network_bias=bias,
    )

    with pytest.raises(ValueError) as ex:
        _ = isogcn.forward(
            phlower_tensor_collection({"h": phlower_tensor(h)}),
            field_data=field,
        )

        assert "Cannot apply nonlinear operator for rank > 0 tensor." in str(ex)


@pytest.mark.parametrize(
    "activations, dropouts, bias",
    [(["tanh"], [], False), ([], [], True), ([], [0.1], False)],
)
@pytest.mark.parametrize(
    "n_nodes, feature_dims, propagations",
    [
        (10, (10, 3, 5), ["convolution"]),
        (5, (5, 3, 3, 4), ["convolution"]),
        (10, (10, 3, 3, 3, 4), ["convolution"]),
    ],
)
def test__forbid_operation_for_tensor_with_ah_w(
    activations: list[str],
    dropouts: list[float],
    bias: bool,
    n_nodes: int,
    feature_dims: tuple[int],
    propagations: list[str],
):
    location = np.random.rand(n_nodes, 3)
    h = np.random.rand(*feature_dims)

    support_names = ["isoam_x", "isoam_y", "isoam_z"]
    field = SimulationField(
        field_tensors=phlower_tensor_collection(
            generate_isoam_like_matrix(location, support_names)
        )
    )
    isogcn = IsoGCN(
        nodes=[feature_dims[-1], 10],
        mul_order="ah_w",
        isoam_names=support_names,
        propagations=propagations,
        use_self_network=True,
        self_network_activations=activations,
        self_network_dropouts=dropouts,
        self_network_bias=bias,
    )

    with pytest.raises(ValueError) as ex:
        _ = isogcn.forward(
            phlower_tensor_collection({"h": phlower_tensor(h)}),
            field_data=field,
        )

        assert "Cannot apply nonlinear operator for rank > 0 tensor." in str(ex)


# endregion

# region Test for Neumann condition


@pytest.mark.parametrize("n_nodes, n_feature", [(5, 2), (10, 3)])
def test__can_forward_with_neumann_condition(n_nodes: int, n_feature: int):
    # common
    propagations = ["convolution"]

    # base coordination
    location = np.random.rand(n_nodes, 3)
    h = np.random.rand(n_nodes, n_feature)
    neumann = np.random.rand(n_nodes, 3, n_feature)

    support_names = ["isoam_x", "isoam_y", "isoam_z"]
    inversed_moment_name = "inv"
    dict_data = generate_isoam_like_matrix(location, support_names)
    # NOTE: This inverse moment is dummy data
    dict_data.update(
        {inversed_moment_name: phlower_tensor(np.random.rand(n_nodes, 3, 3))}
    )
    field = SimulationField(field_tensors=phlower_tensor_collection(dict_data))

    isogcn = IsoGCN(
        nodes=[n_feature, 10],
        isoam_names=support_names,
        propagations=propagations,
        use_self_network=True,
        self_network_activations=["identity"],
        self_network_dropouts=[],
        self_network_bias=False,
        use_neumann=True,
        neumann_input_name="neumann",
        inversed_moment_name=inversed_moment_name,
    )

    h_res = isogcn.forward(
        phlower_tensor_collection(
            {"h": phlower_tensor(h), "neumann": phlower_tensor(neumann)}
        ),
        field_data=field,
    )
    assert h_res.rank() == 1


def test__forbid_forward_neumann_wihthout_self_network():
    with pytest.raises(ValueError) as ex:
        _ = IsoGCN(
            nodes=[10, 10],
            isoam_names=["isoam_x"],
            propagations=["convolution"],
            use_self_network=False,
            use_neumann=True,
            neumann_input_name="neumann",
            inversed_moment_name="inv",
        )
        assert "Use self_network when neumannn layer" in str(ex)


# NOTE: After integrating graphlow, write neumann boudary

# endregion

# region Test for coefficient network


@pytest.mark.parametrize("n_nodes, n_feature", [(10, 3), (5, 1), (7, 3)])
def test__can_forward_with_coefficient_network(n_nodes: int, n_feature: int):
    # common
    propagations = ["contraction", "contraction"]

    # base coordination
    location = np.random.rand(n_nodes, 3)
    h = np.random.rand(n_nodes, 3, 3, n_feature)

    support_names = ["isoam_x", "isoam_y", "isoam_z"]
    field = SimulationField(
        field_tensors=phlower_tensor_collection(
            generate_isoam_like_matrix(location, support_names)
        )
    )
    isogcn = IsoGCN(
        nodes=[n_feature, 5, 5],
        isoam_names=support_names,
        propagations=propagations,
        use_self_network=True,
        self_network_activations=["identity"],
        use_coefficient=True,
        coefficient_activations=["tanh", "tanh"],
    )
    h_res = isogcn.forward(
        phlower_tensor_collection({"h": phlower_tensor(h)}),
        field_data=field,
    )

    assert h_res.rank() == 0


# endregion
