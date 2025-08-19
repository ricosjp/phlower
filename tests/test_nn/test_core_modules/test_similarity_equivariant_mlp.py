import numpy as np
import pytest
import torch
from phlower import PhlowerTensor, phlower_tensor
from phlower._base._batch import GraphBatchInfo
from phlower._base._dimension import PhysicalDimensions
from phlower._base._functionals import to_batch, unbatch
from phlower._fields import SimulationField
from phlower.collections import phlower_tensor_collection
from phlower.nn import SimilarityEquivariantMLP
from phlower.nn._functionals import _functions
from phlower.utils.exceptions import (
    PhlowerDimensionRequiredError,
    PhlowerInvalidArgumentsError,
)
from scipy.stats import ortho_group


def relative_rmse(actual: np.ndarray, desired: np.ndarray) -> float:
    return (
        np.mean((actual - desired) ** 2) ** 0.5
        / np.mean(desired**2 + 1.0e-8) ** 0.5
    )


def create_field_data(dict_tensor: dict[str, PhlowerTensor]) -> SimulationField:
    return SimulationField(phlower_tensor_collection(dict_tensor))


def assert_similarity_equivariance(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    activation: str,
    n_output_feature: int,
    dimension: dict,
    norm_function_name: str,
    centering: bool,
    cross_interaction: bool,
    normalize: bool,
    variable_length_scale: bool,
):
    orthogonal_tensor = PhlowerTensor(
        torch.tensor(ortho_group.rvs(3).astype(np.float32))
    )
    dict_scaling_factor = {
        k: np.random.rand() * 10 + 0.1 for k, v in dimension.items()
    }
    scaling_factor = np.prod(
        [dict_scaling_factor[k] ** v for k, v in dimension.items()]
    )

    create_linear_weight = size[-1] != n_output_feature
    model = SimilarityEquivariantMLP(
        nodes=[size[-1], n_output_feature, n_output_feature],
        scale_names={"T": "T", "L": "L", "M": "M"},
        activations=[activation, activation],
        create_linear_weight=create_linear_weight,
        norm_function_name=norm_function_name,
        disable_en_equivariance=False,
        centering=centering,
        cross_interaction=cross_interaction,
        normalize=normalize,
        unbatch=False,
    )

    t = phlower_tensor(
        torch.rand(*size) * 2 - 1.0,
        dimension=dimension,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
    )
    dict_tensor = {"tensor": t}
    dict_scales = {
        k: phlower_tensor(
            torch.rand(1) * 10 + 0.1, dimension=PhysicalDimensions({k: 1})
        )
        for k, v in dimension.items()
    }
    if variable_length_scale:
        # Length scale differs depending on vertices
        if is_voxel:
            n = round(t.n_vertices() ** (1 / 3))
            dict_scales["L"] = phlower_tensor(
                torch.rand(n, n, n, 1) * 2 + 1.0,
                is_voxel=is_voxel,
                is_time_series=False,
                dimension=PhysicalDimensions({"L": 1}),
            )
        else:
            dict_scales["L"] = phlower_tensor(
                torch.rand(t.n_vertices(), 1) * 2 + 1.0,
                is_voxel=is_voxel,
                is_time_series=False,
                dimension=PhysicalDimensions({"L": 1}),
            )
    field_data = create_field_data(dict_scales)

    ts = phlower_tensor_collection(dict_tensor)
    actual_tensor = (
        _functions.apply_orthogonal_group(
            orthogonal_tensor, model(ts, field_data=field_data)
        )
        * scaling_factor
    )
    actual = actual_tensor.to_numpy()

    assert not np.any(np.isnan(actual))

    dict_transformed = {
        "tensor": _functions.apply_orthogonal_group(orthogonal_tensor, t)
        * scaling_factor
    }
    scaled_field_data = create_field_data(
        {k: v * dict_scaling_factor[k] for k, v in field_data.items()}
    )

    transformed_phlower_tensors = phlower_tensor_collection(dict_transformed)
    desired = model(
        transformed_phlower_tensors, field_data=scaled_field_data
    ).to_numpy()

    # Test equivariance
    assert relative_rmse(actual, desired) < 5.0e-3

    # Test dimension is kept
    for k, v in actual_tensor.dimension.to_dict().items():
        np.testing.assert_almost_equal(v, dimension[k])


def test__can_call_parameters():
    model = SimilarityEquivariantMLP(
        nodes=[8, 8],
        scale_names={"T": "T", "L": "L", "M": "M"},
    )

    # To check Concatenator inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize(
    "size, is_time_series, is_voxel",
    [
        ((10, 1), False, False),
        ((10, 16), False, False),
        ((10, 3, 16), False, False),
        ((4, 10, 1), True, False),
        ((4, 10, 16), True, False),
        ((4, 10, 3, 16), True, False),
        ((10, 10, 10, 1), False, True),
        ((10, 10, 10, 16), False, True),
        ((10, 10, 10, 3, 16), False, True),
        ((4, 10, 10, 10, 1), True, True),
        ((4, 10, 10, 10, 16), True, True),
        ((4, 10, 10, 10, 3, 16), True, True),
    ],
)
@pytest.mark.parametrize("activation", ["identity", "tanh", "leaky_relu0p5"])
@pytest.mark.parametrize("n_output_feature", [1, 16, 32])
@pytest.mark.parametrize(
    "dimension",
    [
        # Dimensionless
        {"T": 0, "L": 0, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Velocity
        {"T": -1, "L": 1, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Mass density
        {"T": 0, "L": -3, "M": 1, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Momentum density
        {"T": -1, "L": 1 - 3, "M": 1, "I": 0, "Theta": 0, "N": 0, "J": 0},
    ],
)
@pytest.mark.parametrize("centering", [False, True])
@pytest.mark.parametrize("cross_interaction", [False])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("variable_length_scale", [False, True])
def test__similarity_equivariance_wo_cross_interaction(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    activation: str,
    n_output_feature: int,
    dimension: dict,
    centering: bool,
    cross_interaction: bool,
    normalize: bool,
    variable_length_scale: bool,
):
    assert_similarity_equivariance(
        size=size,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
        activation=activation,
        n_output_feature=n_output_feature,
        dimension=dimension,
        norm_function_name="identity",
        centering=centering,
        cross_interaction=cross_interaction,
        normalize=normalize,
        variable_length_scale=variable_length_scale,
    )


@pytest.mark.parametrize(
    "size, is_time_series, is_voxel",
    [
        ((10, 1), False, False),
        ((10, 16), False, False),
        ((10, 3, 16), False, False),
        ((4, 10, 1), True, False),
        ((4, 10, 16), True, False),
        ((4, 10, 3, 16), True, False),
        ((10, 10, 10, 1), False, True),
        ((10, 10, 10, 16), False, True),
        ((10, 10, 10, 3, 16), False, True),
        ((4, 10, 10, 10, 1), True, True),
        ((4, 10, 10, 10, 16), True, True),
        ((4, 10, 10, 10, 3, 16), True, True),
    ],
)
@pytest.mark.parametrize("activation", ["identity", "tanh", "leaky_relu0p5"])
@pytest.mark.parametrize("n_output_feature", [1, 16, 32])
@pytest.mark.parametrize(
    "dimension",
    [
        # Dimensionless
        {"T": 0, "L": 0, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Velocity
        {"T": -1, "L": 1, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Mass density
        {"T": 0, "L": -3, "M": 1, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Momentum density
        {"T": -1, "L": 1 - 3, "M": 1, "I": 0, "Theta": 0, "N": 0, "J": 0},
    ],
)
@pytest.mark.parametrize("norm_function_name", ["identity"])
@pytest.mark.parametrize("centering", [False, True])
@pytest.mark.parametrize("cross_interaction", [True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("variable_length_scale", [False, True])
def test__similarity_equivariance_w_cross_interaction(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    activation: str,
    n_output_feature: int,
    dimension: dict,
    norm_function_name: str,
    centering: bool,
    cross_interaction: bool,
    normalize: bool,
    variable_length_scale: bool,
):
    assert_similarity_equivariance(
        size=size,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
        activation=activation,
        n_output_feature=n_output_feature,
        dimension=dimension,
        norm_function_name=norm_function_name,
        centering=centering,
        cross_interaction=cross_interaction,
        normalize=normalize,
        variable_length_scale=variable_length_scale,
    )


@pytest.mark.parametrize(
    "size, is_time_series, is_voxel",
    [
        ((10, 1), False, False),
        ((10, 16), False, False),
        ((10, 3, 16), False, False),
        ((4, 10, 1), True, False),
        ((4, 10, 16), True, False),
        ((4, 10, 3, 16), True, False),
        ((10, 10, 10, 1), False, True),
        ((10, 10, 10, 16), False, True),
        ((10, 10, 10, 3, 16), False, True),
        ((4, 10, 10, 10, 1), True, True),
        ((4, 10, 10, 10, 16), True, True),
        ((4, 10, 10, 10, 3, 16), True, True),
    ],
)
@pytest.mark.parametrize("activation", ["identity", "tanh", "leaky_relu0p5"])
@pytest.mark.parametrize("n_output_feature", [1, 16, 32])
@pytest.mark.parametrize(
    "dimension",
    [
        # Dimensionless
        {"T": 0, "L": 0, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Velocity
        {"T": -1, "L": 1, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Mass density
        {"T": 0, "L": -3, "M": 1, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Momentum density
        {"T": -1, "L": 1 - 3, "M": 1, "I": 0, "Theta": 0, "N": 0, "J": 0},
    ],
)
@pytest.mark.parametrize("centering", [False, True])
@pytest.mark.parametrize("cross_interaction", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test__similarity_invariance(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    activation: str,
    n_output_feature: int,
    dimension: dict,
    centering: bool,
    cross_interaction: bool,
    normalize: bool,
):
    orthogonal_tensor = PhlowerTensor(
        torch.tensor(ortho_group.rvs(3).astype(np.float32))
    )
    dict_scaling_factor = {
        k: np.random.rand() * 10 + 0.1 for k, v in dimension.items()
    }
    scaling_factor = np.prod(
        [dict_scaling_factor[k] ** v for k, v in dimension.items()]
    )

    create_linear_weight = size[-1] != n_output_feature
    model = SimilarityEquivariantMLP(
        nodes=[size[-1], n_output_feature, n_output_feature],
        scale_names={"T": "T", "L": "L", "M": "M"},
        activations=[activation, activation],
        create_linear_weight=create_linear_weight,
        norm_function_name="identity",
        disable_en_equivariance=False,
        centering=centering,
        cross_interaction=cross_interaction,
        normalize=normalize,
        invariant=True,
        unbatch=False,
    )

    t = phlower_tensor(
        torch.rand(*size) * 2 - 1.0,
        dimension=dimension,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
    )
    dict_tensor = {"tensor": t}
    field_data = create_field_data(
        {
            k: phlower_tensor(
                torch.rand(1) * 10 + 0.1, dimension=PhysicalDimensions({k: 1})
            )
            for k, v in dimension.items()
        }
    )

    ts = phlower_tensor_collection(dict_tensor)
    actual_tensor = _functions.apply_orthogonal_group(
        orthogonal_tensor, model(ts, field_data=field_data)
    )
    actual = actual_tensor.to_numpy()

    dict_transformed = {
        "tensor": _functions.apply_orthogonal_group(orthogonal_tensor, t)
        * scaling_factor
    }
    scaled_field_data = create_field_data(
        {k: v * dict_scaling_factor[k] for k, v in field_data.items()}
    )

    transformed_phlower_tensors = phlower_tensor_collection(dict_transformed)
    desired = model(
        transformed_phlower_tensors, field_data=scaled_field_data
    ).to_numpy()

    # Test equivariance
    assert relative_rmse(actual, desired) < 5.0e-3

    # Test dimensionless in case of invariant
    for v in actual_tensor.dimension.to_dict().values():
        np.testing.assert_almost_equal(v, 0.0)


@pytest.mark.parametrize(
    "size, is_time_series, is_voxel",
    [
        ((10, 1), False, False),
        ((10, 16), False, False),
        ((10, 3, 16), False, False),
        ((4, 10, 1), True, False),
        ((4, 10, 16), True, False),
        ((4, 10, 3, 16), True, False),
        ((10, 10, 10, 1), False, True),
        ((10, 10, 10, 16), False, True),
        ((10, 10, 10, 3, 16), False, True),
        ((4, 10, 10, 10, 1), True, True),
        ((4, 10, 10, 10, 16), True, True),
        ((4, 10, 10, 10, 3, 16), True, True),
    ],
)
@pytest.mark.parametrize("activation", ["identity", "tanh", "leaky_relu0p5"])
@pytest.mark.parametrize("n_output_feature", [1, 16, 32])
@pytest.mark.parametrize(
    "dimension",
    [
        # Dimensionless
        {"T": 0, "L": 0, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Velocity
        {"T": -1, "L": 1, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Mass density
        {"T": 0, "L": -3, "M": 1, "I": 0, "Theta": 0, "N": 0, "J": 0},
        # Momentum density
        {"T": -1, "L": 1 - 3, "M": 1, "I": 0, "Theta": 0, "N": 0, "J": 0},
    ],
)
@pytest.mark.parametrize("centering", [False, True])
@pytest.mark.parametrize("cross_interaction", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("invariant", [False, True])
def test__scaling_equivariance(
    size: tuple[int],
    is_time_series: bool,
    is_voxel: bool,
    activation: str,
    n_output_feature: int,
    dimension: dict,
    centering: bool,
    cross_interaction: bool,
    normalize: bool,
    invariant: bool,
):
    orthogonal_tensor = PhlowerTensor(torch.eye(3))
    dict_scaling_factor = {
        k: np.random.rand() * 10 + 0.1 for k, v in dimension.items()
    }
    scaling_factor = np.prod(
        [dict_scaling_factor[k] ** v for k, v in dimension.items()]
    )

    create_linear_weight = size[-1] != n_output_feature
    model = SimilarityEquivariantMLP(
        nodes=[size[-1], n_output_feature, n_output_feature],
        scale_names={"T": "T", "L": "L", "M": "M"},
        activations=[activation, activation],
        create_linear_weight=create_linear_weight,
        norm_function_name="identity",
        disable_en_equivariance=True,
        centering=centering,
        cross_interaction=cross_interaction,
        normalize=normalize,
        invariant=invariant,
        unbatch=False,
    )

    t = phlower_tensor(
        torch.rand(*size) * 2 - 1.0,
        dimension=dimension,
        is_time_series=is_time_series,
        is_voxel=is_voxel,
    )
    dict_tensor = {"tensor": t}
    field_data = create_field_data(
        {
            k: phlower_tensor(
                torch.rand(1) * 10 + 0.1, dimension=PhysicalDimensions({k: 1})
            )
            for k, v in dimension.items()
        }
    )

    ts = phlower_tensor_collection(dict_tensor)
    if invariant:
        actual_tensor = (
            _functions.apply_orthogonal_group(
                orthogonal_tensor, model(ts, field_data=field_data)
            )
            * 1.0
        )
    else:
        actual_tensor = (
            _functions.apply_orthogonal_group(
                orthogonal_tensor, model(ts, field_data=field_data)
            )
            * scaling_factor
        )
    actual = actual_tensor.to_numpy()

    dict_transformed = {
        "tensor": _functions.apply_orthogonal_group(orthogonal_tensor, t)
        * scaling_factor
    }
    scaled_field_data = create_field_data(
        {k: v * dict_scaling_factor[k] for k, v in field_data.items()}
    )

    transformed_phlower_tensors = phlower_tensor_collection(dict_transformed)
    desired = model(
        transformed_phlower_tensors, field_data=scaled_field_data
    ).to_numpy()

    # Test equivariance
    assert relative_rmse(actual, desired) < 5.0e-3

    if invariant:
        # Test dimensionless in case of invariant
        for v in actual_tensor.dimension.to_dict().values():
            np.testing.assert_almost_equal(v, 0.0)
    else:
        # Test dimension is kept
        for k, v in actual_tensor.dimension.to_dict().items():
            np.testing.assert_almost_equal(v, dimension[k])


def test__similarity_equivariance_no_dimension():
    model = SimilarityEquivariantMLP(
        nodes=[8, 8],
        scale_names={"T": "T"},
    )

    t = phlower_tensor(torch.rand(10, 3, 8), dimension=None)
    t_scale = phlower_tensor(
        torch.tensor(0.1), dimension=PhysicalDimensions({"T": 1})
    )
    field_data = create_field_data({"T": t_scale})
    ts = phlower_tensor_collection({"tensor": t})
    with pytest.raises(PhlowerDimensionRequiredError):
        model(ts, field_data=field_data)


def test__similarity_equivariance_no_scale_input():
    model = SimilarityEquivariantMLP(
        nodes=[8, 8],
        scale_names={"T": "T", "L": "L", "M": "M"},
        unbatch=False,
    )
    dimension = {"T": -1, "L": 1, "M": 0, "I": 0, "Theta": 0, "N": 0, "J": 0}

    t = phlower_tensor(torch.rand(10, 3, 8), dimension=dimension)
    ts = phlower_tensor_collection({"tensor": t})
    with pytest.raises(PhlowerInvalidArgumentsError):
        model(ts)


@pytest.mark.parametrize("shape", [(4, 16), (4, 3, 16)])
@pytest.mark.parametrize("dimension", [{}, {"L": 1, "T": 1}, {"M": 1}])
def test__batch_correct(
    shape: tuple[int],
    dimension: dict[str, int],
):
    model = SimilarityEquivariantMLP(
        nodes=[16, 16, 16],
        scale_names={"T": "T", "L": "L", "M": "M"},
        activations=["tanh", "tanh"],
    )

    variable1 = phlower_tensor(torch.rand(shape), dimension=dimension)
    variable2 = phlower_tensor(torch.rand(shape), dimension=dimension)

    n = shape[0]
    dict_scales1 = {
        k: phlower_tensor(
            torch.rand(1) * 10 + 0.1, dimension=PhysicalDimensions({k: 1})
        )
        for k in model._scale_names.values()
    }
    dict_scales1["L"] = phlower_tensor(
        torch.rand(n, 1) * 2 + 1.0,
        dimension=PhysicalDimensions({"L": 1}),
    )
    dict_scales2 = {
        k: phlower_tensor(
            torch.rand(1) * 10 + 0.1, dimension=PhysicalDimensions({k: 1})
        )
        for k in model._scale_names.values()
    }
    dict_scales2["L"] = phlower_tensor(
        torch.rand(n, 1) * 2 + 1.0,
        dimension=PhysicalDimensions({"L": 1}),
    )

    batched_variable, _ = to_batch([variable1, variable2])
    batched_dict_scales = {
        k: to_batch([dict_scales1[k], dict_scales2[k]])[0]
        for k in dict_scales1.keys()
    }
    dict_batch_info = {
        k: to_batch([dict_scales1[k], dict_scales2[k]])[1]
        for k in dict_scales1.keys()
    }
    ts = phlower_tensor_collection({"t": batched_variable})
    field = SimulationField(
        phlower_tensor_collection(batched_dict_scales),
        batch_info=dict_batch_info,
    )
    batched_prediction = model(ts, field_data=field)

    dict_batch_info1 = {
        k: GraphBatchInfo(
            sizes=[v.size],
            shapes=[v.shape],
            n_nodes=[v.n_vertices()],
            dense_concat_dim=0,
        )
        for k, v in dict_scales1.items()
    }
    field1 = SimulationField(
        phlower_tensor_collection(dict_scales1),
        batch_info=dict_batch_info1,
    )
    pred1 = model(
        phlower_tensor_collection({"t": variable1}),
        field_data=field1,
    )

    dict_batch_info2 = {
        k: GraphBatchInfo(
            sizes=[v.size],
            shapes=[v.shape],
            n_nodes=[v.n_vertices()],
            dense_concat_dim=0,
        )
        for k, v in dict_scales2.items()
    }
    field2 = SimulationField(
        phlower_tensor_collection(dict_scales2),
        batch_info=dict_batch_info2,
    )
    pred2 = model(
        phlower_tensor_collection({"t": variable2}),
        field_data=field2,
    )

    unbatched_pred = unbatch(
        batched_prediction, batch_info=dict_batch_info["L"]
    )
    np.testing.assert_almost_equal(
        unbatched_pred[0].numpy(), pred1.numpy(), decimal=5
    )
    np.testing.assert_almost_equal(
        unbatched_pred[1].numpy(), pred2.numpy(), decimal=5
    )
