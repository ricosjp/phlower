import pathlib

import numpy as np
import pytest
import torch
from phlower_tensor import phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn._phlower_module_adapter import PhlowerModuleAdapter
from phlower.settings import ModuleSetting
from phlower.settings._debug_parameter_setting import (
    PhlowerModuleDebugParameters,
)
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
