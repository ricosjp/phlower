import numpy as np
import pytest
import torch
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import phlower_tensor_collection
from scipy.stats import ortho_group

from phlower.nn import Slip


def _random_unit_normals(n_nodes: int) -> np.ndarray:
    normals = np.random.rand(n_nodes, 3, 1) - 0.5
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _random_flags(n_nodes: int) -> np.ndarray:
    flags = (np.random.rand(n_nodes, 1) > 0.5).astype(float)
    flags[0] = 1.0  # ensure both values appear
    flags[1] = 0.0
    return flags


def test__can_call_parameters():
    model = Slip("identity", normal_name="", flag_name="")

    # To check Slip inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize("n_feature", [1, 4])
def test__removes_normal_component_only_on_flagged_nodes(n_feature: int):
    n_nodes = 10
    velocity = np.random.rand(n_nodes, 3, n_feature)
    normals = _random_unit_normals(n_nodes)
    flags = _random_flags(n_nodes)

    phlower_tensors = phlower_tensor_collection(
        {
            "velocity": phlower_tensor(torch.from_numpy(velocity)),
            "normal": phlower_tensor(torch.from_numpy(normals)),
            "flag": phlower_tensor(torch.from_numpy(flags)),
        }
    )

    model = Slip("identity", normal_name="normal", flag_name="flag")

    actual = model(phlower_tensors).to_numpy()

    desired = velocity - np.einsum(
        "npf,nph,nqh,ng->nqf", velocity, normals, normals, flags
    )
    np.testing.assert_almost_equal(desired, actual)

    dots = np.einsum("npf,nph->nf", actual, normals)
    on_slip = flags[:, 0] == 1.0
    np.testing.assert_almost_equal(dots[on_slip], 0.0)
    np.testing.assert_almost_equal(actual[~on_slip], velocity[~on_slip])


def test__rotation_equivariance():
    n_nodes = 10
    n_feature = 4
    velocity = np.random.rand(n_nodes, 3, n_feature)
    normals = _random_unit_normals(n_nodes)
    flags = _random_flags(n_nodes)

    rotation = ortho_group.rvs(3)

    model = Slip("identity", normal_name="normal", flag_name="flag")

    def _apply(velocity: np.ndarray, normals: np.ndarray) -> np.ndarray:
        phlower_tensors = phlower_tensor_collection(
            {
                "velocity": phlower_tensor(torch.from_numpy(velocity)),
                "normal": phlower_tensor(torch.from_numpy(normals)),
                "flag": phlower_tensor(torch.from_numpy(flags)),
            }
        )
        return model(phlower_tensors).to_numpy()

    actual = _apply(
        np.einsum("pq,nqf->npf", rotation, velocity),
        np.einsum("pq,nqh->nph", rotation, normals),
    )
    desired = np.einsum("pq,nqf->npf", rotation, _apply(velocity, normals))

    np.testing.assert_almost_equal(desired, actual)


def test__time_series_value():
    n_time = 5
    n_nodes = 10
    n_feature = 4
    velocity = np.random.rand(n_time, n_nodes, 3, n_feature)
    normals = _random_unit_normals(n_nodes)
    flags = _random_flags(n_nodes)

    phlower_tensors = phlower_tensor_collection(
        {
            "velocity": phlower_tensor(
                torch.from_numpy(velocity), is_time_series=True
            ),
            "normal": phlower_tensor(torch.from_numpy(normals)),
            "flag": phlower_tensor(torch.from_numpy(flags)),
        }
    )

    model = Slip("identity", normal_name="normal", flag_name="flag")

    actual = model(phlower_tensors)

    assert actual.is_time_series

    desired = velocity - np.einsum(
        "tnpf,nph,nqh,ng->tnqf", velocity, normals, normals, flags
    )
    np.testing.assert_almost_equal(desired, actual.to_numpy())


@pytest.mark.parametrize("time_series", [False, True])
def test__normal_features_broadcast_over_value_features(time_series: bool):
    n_nodes = 10
    n_feature = 5
    shape = (
        (4, n_nodes, 3, n_feature) if time_series else (n_nodes, 3, n_feature)
    )
    velocity = np.random.rand(*shape)
    normals = _random_unit_normals(n_nodes)
    flags = _random_flags(n_nodes)

    model = Slip("identity", normal_name="normal", flag_name="flag")

    def _apply(velocity: np.ndarray) -> np.ndarray:
        phlower_tensors = phlower_tensor_collection(
            {
                "velocity": phlower_tensor(
                    torch.from_numpy(velocity), is_time_series=time_series
                ),
                "normal": phlower_tensor(torch.from_numpy(normals)),
                "flag": phlower_tensor(torch.from_numpy(flags)),
            }
        )
        return model(phlower_tensors).to_numpy()

    actual = _apply(velocity)

    # Check that the single-feature normal broadcasts correctly and
    # applies the same projection to every feature of the velocity.
    for i in range(n_feature):
        np.testing.assert_almost_equal(
            actual[..., [i]], _apply(velocity[..., [i]])
        )


def test__raise_error_when_value_cannot_be_detected():
    n_nodes = 10
    phlower_tensors = phlower_tensor_collection(
        {
            "velocity": phlower_tensor(torch.rand(n_nodes, 3, 4)),
            "pressure": phlower_tensor(torch.rand(n_nodes, 1)),
            "normal": phlower_tensor(torch.rand(n_nodes, 3, 1)),
            "flag": phlower_tensor(torch.rand(n_nodes, 1)),
        }
    )

    model = Slip("identity", normal_name="normal", flag_name="flag")

    with pytest.raises(ValueError) as ex:
        model(phlower_tensors)

    assert "not unique" in str(ex.value)


@pytest.mark.parametrize(
    ("normal_shape", "flag_shape"),
    [((10, 3, 4), (10, 1)), ((10, 3, 1), (10, 4))],
)
def test__raise_error_when_normal_or_flag_has_multiple_features(
    normal_shape: tuple[int, ...], flag_shape: tuple[int, ...]
):
    phlower_tensors = phlower_tensor_collection(
        {
            "velocity": phlower_tensor(torch.rand(10, 3, 4)),
            "normal": phlower_tensor(torch.rand(*normal_shape)),
            "flag": phlower_tensor(torch.rand(*flag_shape)),
        }
    )

    model = Slip("identity", normal_name="normal", flag_name="flag")

    with pytest.raises(ValueError) as ex:
        model(phlower_tensors)

    assert "feature dimension of normal and flag must be 1" in str(ex.value)


def test__raise_error_when_value_is_not_rank_1():
    n_nodes = 10
    phlower_tensors = phlower_tensor_collection(
        {
            "pressure": phlower_tensor(torch.rand(n_nodes, 4)),
            "normal": phlower_tensor(torch.rand(n_nodes, 3, 1)),
            "flag": phlower_tensor(torch.rand(n_nodes, 1)),
        }
    )

    model = Slip("identity", normal_name="normal", flag_name="flag")

    with pytest.raises(ValueError) as ex:
        model(phlower_tensors)

    assert "only applicable to a rank-1 tensor" in str(ex.value)
