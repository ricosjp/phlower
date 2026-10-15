import numpy as np
import pytest
import torch
from phlower_tensor import SimulationField, phlower_tensor
from phlower_tensor.collections import phlower_tensor_collection
from phlower_tensor.functionals import to_batch

from phlower.nn import MeanConstrainedSigmoid
from phlower.settings._module_settings import MeanConstrainedSigmoidSetting


def test__can_call_parameters():
    model = MeanConstrainedSigmoid(
        output_feature_dim=10,
        reference_variable_name="a",
    )

    # To check Einsum inherit torch.nn.Module appropriately
    _ = model.parameters()


@pytest.mark.parametrize("output_feature_dim", [10, 20])
@pytest.mark.parametrize("reference_variable_name", ["a", "b"])
@pytest.mark.parametrize("weight_field_name", [None, "w"])
@pytest.mark.parametrize("slope", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("learnable_slope", [True, False])
@pytest.mark.parametrize("epsilon", [1.0e-6, 1.0e-5])
def test__can_pass_parameters_via_setting(
    output_feature_dim: int,
    reference_variable_name: str,
    weight_field_name: str | None,
    slope: float,
    learnable_slope: bool,
    epsilon: float,
):
    setting = MeanConstrainedSigmoidSetting(
        output_feature_dim=output_feature_dim,
        reference_variable_name=reference_variable_name,
        weight_field_name=weight_field_name,
        slope=slope,
        learnable_slope=learnable_slope,
        epsilon=epsilon,
    )
    model = MeanConstrainedSigmoid.from_setting(setting)

    assert model._output_feature_dim == output_feature_dim
    assert model._reference_variable_name == reference_variable_name

    assert model._weight_field_name == weight_field_name
    assert model._learnable_slope == learnable_slope
    if not model._learnable_slope:
        np.testing.assert_array_almost_equal(model._slope.numpy(), slope)
    assert model._epsilon == epsilon


@pytest.mark.parametrize("n_nodes", [100, 200])
@pytest.mark.parametrize("output_feature_dim", [1, 4, 8])
@pytest.mark.parametrize("slope", [0.1, 1.0, 10.0, 100.0])
@pytest.mark.parametrize("weight_field_name", [None, "w"])
@pytest.mark.parametrize("learnable_slope", [True, False])
def test__target_mean_correct(
    n_nodes: int,
    output_feature_dim: int,
    slope: float,
    weight_field_name: str | None,
    learnable_slope: bool,
):
    ref = torch.sigmoid(phlower_tensor(torch.randn(n_nodes, 1) + torch.rand(1)))
    feature = phlower_tensor(
        torch.randn(n_nodes, output_feature_dim)
        + torch.randn(output_feature_dim) * 10
    )

    phc = phlower_tensor_collection({"ref": ref, "feature": feature})
    _, batch_info = to_batch([ref])
    w = phlower_tensor(torch.rand(n_nodes, 1)) + 0.1
    field_data = SimulationField({"w": w}, {"sample": batch_info})

    mcs = MeanConstrainedSigmoid(
        output_feature_dim=output_feature_dim,
        reference_variable_name="ref",
        weight_field_name=weight_field_name,
        slope=slope,
        learnable_slope=learnable_slope,
    )
    out = mcs(phc, field_data=field_data).detach().numpy()
    if weight_field_name is None:
        np.testing.assert_allclose(
            np.mean(out, axis=0), np.mean(ref.numpy()), rtol=1e-5
        )
    else:
        np.testing.assert_allclose(
            np.einsum("ng,nf->f", w, out) / n_nodes,
            np.einsum("ng,nf->", w, ref.numpy()) / n_nodes,
            rtol=1e-5,
        )
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


@pytest.mark.parametrize("n_nodes", [200])
@pytest.mark.parametrize("output_feature_dim", [8])
@pytest.mark.parametrize("slope", [10.0])
@pytest.mark.parametrize("learnable_slope", [True])
@pytest.mark.parametrize("offset_value", [-1.0, 2.0])
def test__raise_when_infeasible(
    n_nodes: int,
    output_feature_dim: int,
    slope: float,
    learnable_slope: bool,
    offset_value: float,
):
    ref = torch.sigmoid(phlower_tensor(torch.randn(n_nodes, 1))) + offset_value
    feature = phlower_tensor(
        torch.randn(n_nodes, output_feature_dim)
        + torch.randn(output_feature_dim) * 10
    )

    phc = phlower_tensor_collection({"ref": ref, "feature": feature})
    mcs = MeanConstrainedSigmoid(
        output_feature_dim=output_feature_dim,
        reference_variable_name="ref",
        slope=slope,
        learnable_slope=learnable_slope,
    )
    with pytest.raises(ValueError, match="infeasible"):
        mcs(phc)
