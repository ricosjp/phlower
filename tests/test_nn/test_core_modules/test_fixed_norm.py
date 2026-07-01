import hypothesis.strategies as st
import pytest
import torch
from hypothesis import given
from phlower_tensor import PhlowerTensor
from phlower_tensor.collections import phlower_tensor_collection

from phlower.nn import FixedNorm
from phlower.settings._module_settings import FixedNormSetting
from phlower.utils import create_simulation_field


def test__can_call_parameters():
    model = FixedNorm(nodes=[10, 10])

    # To check FixedNorm inherit torch.nn.Module appropriately
    _ = model.parameters()


@given(
    nodes=st.integers(min_value=1, max_value=10).map(lambda x: [x, x]),
    mean_name=st.text(min_size=1, max_size=10),
    std_name=st.text(min_size=1, max_size=10),
)
def test__can_pass_parameters_via_setting(
    nodes: list[int],
    mean_name: str,
    std_name: str,
):
    setting = FixedNormSetting(
        nodes=nodes,
        mean_name=mean_name,
        std_name=std_name,
    )
    model = FixedNorm.from_setting(setting)

    assert model._nodes == nodes
    assert model._mean_name == mean_name
    assert model._std_name == std_name


@pytest.mark.parametrize(
    "input_shape",
    [
        (10, 1),
        (10, 16),
        (10, 3, 16),
    ],
)
def test__layer_norm(input_shape: tuple[int]):
    ph_tensor = PhlowerTensor(torch.rand(*input_shape))
    phlower_tensors = phlower_tensor_collection({"tensor": ph_tensor})

    _mean = torch.mean(ph_tensor, dim=0, keepdim=True)
    _std = torch.abs(torch.std(ph_tensor, dim=0, keepdim=True)) + 1e-5
    field_data = create_simulation_field(
        field_tensors={"mean": _mean, "std": _std}, batch_info={}
    )

    model = FixedNorm(mean_name="mean", std_name="std")
    actual: PhlowerTensor = model(phlower_tensors, field_data=field_data)

    assert actual.shape == input_shape

    desired = (ph_tensor - _mean) / _std
    assert torch.allclose(actual.to_tensor(), desired.to_tensor())


@pytest.mark.parametrize(
    "mean_name, std_name",
    [
        ("mean", "wrong_std"),
        ("wrong_mean", "std"),
        ("wrong_mean", "wrong_std"),
    ],
)
def test__missing_mean_or_std(
    mean_name: str,
    std_name: str,
):
    ph_tensor = PhlowerTensor(torch.rand(10, 3))
    phlower_tensors = phlower_tensor_collection({"tensor": ph_tensor})

    _mean = torch.mean(ph_tensor, dim=0)
    _std = torch.abs(torch.std(ph_tensor, dim=0)) + 1e-5
    field_data = create_simulation_field(
        field_tensors={"mean": _mean, "std": _std}, batch_info={}
    )

    model = FixedNorm(mean_name=mean_name, std_name=std_name)
    with pytest.raises(KeyError, match="Mean or std not found in field_data."):
        _ = model(phlower_tensors, field_data=field_data)
