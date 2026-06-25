import pathlib
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from phlower_tensor import phlower_array, phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn import GCN
from phlower.nn._phlower_module_adapter import PhlowerModuleAdapter
from phlower.settings import ModuleSetting
from phlower.settings._debug_parameter_setting import (
    PhlowerModuleDebugParameters,
)
from phlower.utils import create_simulation_field
from phlower.utils.exceptions import PhlowerRunTimeError
from phlower.utils.typing import CalculationState


@pytest.mark.parametrize("coeff", [1.0, 2.0, -3.2])
def test__coeff_factor_with_identity_module(coeff: float):
    setting = ModuleSetting(
        nn_type="Identity", name="aa", input_keys=["sample"], coeff=coeff
    )
    sample_input = phlower_tensor(torch.rand(2, 3))
    inputs = phlower_tensor_collection({"sample": sample_input})

    model = PhlowerModuleAdapter.from_setting(setting)
    actual = model.forward(inputs).unique_item()

    np.testing.assert_array_almost_equal(
        actual.to_tensor(), sample_input.to_tensor() * coeff
    )


@pytest.mark.parametrize(
    "output_tensor_shape", [(-1, 1), (2, 3, 2), (2, 3, 4, -1)]
)
def test__raise_error_invalid_output_tensor_shape(
    output_tensor_shape: list[int],
):
    debug_parameters = PhlowerModuleDebugParameters(
        output_tensor_shape=output_tensor_shape
    )
    setting = ModuleSetting(
        nn_type="Identity",
        name="aa",
        input_keys=["sample"],
        debug_parameters=debug_parameters,
    )
    input_tensor = phlower_tensor(torch.rand(2, 3, 4))
    input_tensors = phlower_tensor_collection({"sample": input_tensor})
    model = PhlowerModuleAdapter.from_setting(setting)
    with pytest.raises(PhlowerRunTimeError) as ex:
        _ = model.forward(input_tensors)

    assert "is different from desired shape" in str(ex.value)


@pytest.mark.parametrize(
    "output_tensor_shape", [(-1, -1, 4), (2, 3, 4), (2, -1, 4)]
)
def test__pass_output_tensor_shape(output_tensor_shape: list[int]):
    debug_parameters = PhlowerModuleDebugParameters(
        output_tensor_shape=output_tensor_shape
    )
    setting = ModuleSetting(
        nn_type="Identity",
        name="aa",
        input_keys=["sample"],
        debug_parameters=debug_parameters,
    )
    input_tensor = phlower_tensor(torch.rand(2, 3, 4))
    input_tensors = phlower_tensor_collection({"sample": input_tensor})
    model = PhlowerModuleAdapter.from_setting(setting)
    _ = model.forward(input_tensors)


@pytest.mark.parametrize(
    "dump_forward_en_route_tensor, dump_backward_en_route_tensor, expected",
    [
        (True, False, ["feat.npy"]),
        (False, True, ["feat_grad_output.npy"]),
        (True, True, ["feat.npy", "feat_grad_output.npy"]),
    ],
)
def test_dump_en_route_tensor(
    dump_forward_en_route_tensor: bool,
    dump_backward_en_route_tensor: bool,
    expected: list[str],
    tmp_path: pathlib.Path,
):
    debug_parameters = PhlowerModuleDebugParameters(
        dump_forward_tensor=dump_forward_en_route_tensor,
        dump_backward_tensor=dump_backward_en_route_tensor,
    )
    setting = ModuleSetting(
        nn_type="MLP",
        name="aa",
        input_keys=["sample"],
        output_key="feat",
        debug_parameters=debug_parameters,
        nn_parameters={"nodes": [10, 20, 5], "activations": ["tanh", "tanh"]},
    )
    input_tensor = phlower_tensor(torch.rand(100, 10, requires_grad=True))
    input_tensors = phlower_tensor_collection({"sample": input_tensor})
    model = PhlowerModuleAdapter.from_setting(setting)
    model.train()

    out: IPhlowerTensorCollections = model(
        input_tensors,
        state=CalculationState(
            mode="training",
            current_epoch=0,
            current_batch_iteration=0,
            output_directory=tmp_path,
        ),
    )
    for v in out.values():
        v._tensor.retain_grad()

    dummy_loss = torch.sum(out.unique_item())
    dummy_loss.backward()
    model.finalize_debug()

    ret = out.unique_item().to_tensor()
    desired = {
        "feat.npy": ret.detach().numpy(),
        "feat_grad_output.npy": ret.grad.detach().numpy()
        if ret.grad is not None
        else None,
    }

    output_files = list((tmp_path / "en_route_tensors").rglob("*.npy"))
    assert len(output_files) == len(expected)
    for target_file in output_files:
        assert target_file.name in expected

        np.testing.assert_array_almost_equal(
            np.load(target_file),
            desired[target_file.name],
        )


def test__field_data_is_overwritten():
    setting = ModuleSetting(
        nn_type="GCN",
        name="aa",
        input_keys=["sample", "support1"],
        output_key="feat",
        nn_parameters={"support_name": "support1", "nodes": [10, 10]},
        input_keys_promoting_to_field=["support1"],
    )
    model = PhlowerModuleAdapter.from_setting(setting)

    support1_field = phlower_tensor(
        phlower_array(
            sp.random(100, 100, density=0.1, format="csr", dtype=np.float32)
        ).to_tensor()
    )
    field_data = create_simulation_field(
        field_tensors={"support1": support1_field},
        batch_info={},
    )

    input_tensor = phlower_tensor(torch.rand(100, 10, requires_grad=True))
    support1_input = phlower_tensor(
        phlower_array(
            sp.random(100, 100, density=0.1, format="csr", dtype=np.float32)
        ).to_tensor()
    )

    # check that the two support tensors are not equal
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(
            support1_field.to_tensor().to_dense().numpy(),
            support1_input.to_tensor().to_dense().numpy(),
        )

    input_tensors = phlower_tensor_collection(
        {"sample": input_tensor, "support1": support1_input}
    )
    model = PhlowerModuleAdapter.from_setting(setting)
    model.train()

    with mock.patch.object(GCN, "forward") as mocked:
        mocked.return_value = phlower_tensor(
            torch.rand(100, 10, requires_grad=True)
        )  # this is dummy

        _ = model.forward(input_tensors, field_data=field_data)
        mocked.assert_called_once()
        _, kwargs = mocked.call_args
        assert "support1" in kwargs["field_data"].keys()
        np.testing.assert_array_almost_equal(
            kwargs["field_data"]["support1"].to_tensor().to_dense().numpy(),
            support1_input.to_tensor().to_dense().numpy(),
        )

    # check that the field data remains unchanged after the forward pass
    np.testing.assert_array_almost_equal(
        field_data["support1"].to_tensor().to_dense().numpy(),
        support1_field.to_tensor().to_dense().numpy(),
    )
