from __future__ import annotations

import pathlib
from copy import deepcopy
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
import torch
import yaml
from jinja2 import Template
from phlower_tensor import ISimulationField, phlower_array, phlower_tensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn import PhlowerGroupModule
from phlower.settings import PhlowerModelSetting
from phlower.utils import create_simulation_field
from phlower.utils.exceptions import PhlowerRunTimeError
from phlower.utils.typing import PhlowerForwardHook

_DATA_DIR = pathlib.Path(__file__).parent / "data/activation_checkpoint"


def _load_model(
    file_path: pathlib.Path, checkpoint: bool, no_grad: bool = False
) -> PhlowerGroupModule:
    template = Template(file_path.read_text(encoding="utf-8"))
    result = template.render(checkpoint=checkpoint, no_grad=no_grad)

    content = yaml.safe_load(result)
    setting = PhlowerModelSetting.model_validate(content["model"])
    setting.resolve()

    model = PhlowerGroupModule.from_setting(setting.network)
    model.resolve()
    return model


def _prepare_inputs() -> tuple[IPhlowerTensorCollections, ISimulationField]:
    n_nodes = 10
    inputs = phlower_tensor_collection(
        {
            "x": phlower_tensor(torch.rand(n_nodes, 3, 2, requires_grad=True)),
            "scalar": phlower_tensor(
                torch.rand(n_nodes, 2, requires_grad=True)
            ),
            "y": phlower_tensor(torch.rand(n_nodes, 3, 2, requires_grad=True)),
            "neumann": phlower_tensor(
                torch.rand(n_nodes, 3, 2, requires_grad=True)
            ),
            "moment": phlower_tensor(
                torch.rand(n_nodes, 3, 3, 1, requires_grad=True)
            ),
        }
    )

    supports = {
        key: phlower_tensor(
            phlower_array(
                sp.random(n_nodes, n_nodes, density=0.1, dtype=np.float32)
            ).to_tensor()
        )
        for key in ["gx", "gy", "gz"]
    }
    field = create_simulation_field(field_tensors=supports, batch_info={})

    return inputs, field


def _detach_clone(
    collection: IPhlowerTensorCollections,
) -> IPhlowerTensorCollections:
    # NOTE: We may need to add this method to IPhlowerTensorCollections
    return phlower_tensor_collection(
        {
            key: phlower_tensor(
                value.to_tensor()
                .detach()
                .clone()
                .requires_grad_(value.requires_grad)
            )
            for key, value in collection.items()
        }
    )


SINGLE_MODULE_FILES = [
    "mlp.yml",
    "en_equivariant_mlp.yml",
    "mlp_w_dropouts.yml",
    "gcn.yml",
    "isogcn.yml",
]
INTEGRATION_MODULE_FILES = [
    "model.yml",
]

single_module_cases = pytest.mark.parametrize("file_name", SINGLE_MODULE_FILES)
integration_module_cases = pytest.mark.parametrize(
    "file_name", INTEGRATION_MODULE_FILES
)
all_module_cases = pytest.mark.parametrize(
    "file_name",
    SINGLE_MODULE_FILES + INTEGRATION_MODULE_FILES,
)


@single_module_cases
@pytest.mark.parametrize(
    "checkpoint, no_grad, desired_calls",
    [
        (False, False, 1),
        (True, False, 2),
        (True, True, 1),
        (False, True, 1),
    ],
)
def test__n_counts_to_call_forward(
    file_name: str, checkpoint: bool, no_grad: bool, desired_calls: int
):
    data_dir = pathlib.Path(__file__).parent / "data/activation_checkpoint"
    model = _load_model(
        data_dir / file_name, checkpoint=checkpoint, no_grad=no_grad
    )
    assert len(model._phlower_modules) == 1

    core_module = model._phlower_modules[0].get_core_module()

    inputs, field_data = _prepare_inputs()
    with mock.patch.object(
        core_module, "forward", wraps=core_module.forward
    ) as mock_forward:
        val: IPhlowerTensorCollections = model(inputs, field_data=field_data)

        sum_val = val.sum()
        assert sum_val.requires_grad is not no_grad
        if not no_grad:
            val.sum().backward()

    assert mock_forward.call_count == desired_calls


@all_module_cases
@pytest.mark.parametrize("seed", [0, 33])
def test__same_output_and_gradients_for_checkpointed_module(
    file_name: str, seed: int
):
    eager_model = _load_model(_DATA_DIR / file_name, checkpoint=False)
    eager_model.train()

    checkpointed_model = _load_model(_DATA_DIR / file_name, checkpoint=True)
    checkpointed_model.train()

    # Ensure both models have the same initial parameters.
    checkpointed_model.load_state_dict(deepcopy(eager_model.state_dict()))

    # --- Run without checkpoint
    inputs, field_data = _prepare_inputs()
    torch.manual_seed(seed)
    eager_output: IPhlowerTensorCollections = eager_model(
        inputs, field_data=field_data
    )
    eager_output.sum().backward()
    eager_rng_state = torch.get_rng_state()
    eager_inputs_grad = {
        key: inputs[key].to_tensor().grad
        for key in inputs.keys()
        if inputs[key].requires_grad
    }

    # --- Run with checkpoint
    inputs = _detach_clone(inputs)
    torch.manual_seed(seed)
    checkpointed_output: IPhlowerTensorCollections = checkpointed_model(
        inputs, field_data=field_data
    )
    checkpointed_output.sum().backward()
    checkpointed_rng_state = torch.get_rng_state()
    checkpointed_inputs_grad = {
        key: inputs[key].to_tensor().grad
        for key in inputs.keys()
        if inputs[key].requires_grad
    }

    # Compare RNG states
    assert torch.equal(eager_rng_state, checkpointed_rng_state)

    # Compare outputs
    assert checkpointed_output.keys() == eager_output.keys()
    for k in checkpointed_output.keys():
        torch.testing.assert_close(
            checkpointed_output[k].to_tensor(),
            eager_output[k].to_tensor(),
            rtol=1e-5,
            atol=1e-8,
        )

    # Compare inputs' gradient
    assert checkpointed_inputs_grad.keys() == eager_inputs_grad.keys()
    for k in checkpointed_inputs_grad.keys():
        torch.testing.assert_close(
            checkpointed_inputs_grad[k],
            eager_inputs_grad[k],
            rtol=1e-5,
            atol=1e-8,
        )

    # Compare parameters' gradients
    eager_params = list(eager_model.named_parameters())
    checkpointed_params = list(checkpointed_model.named_parameters())
    assert len(eager_params) == len(checkpointed_params)

    for (_, eager_param), (_, checkpoint_param) in zip(
        eager_params, checkpointed_params, strict=True
    ):
        assert eager_param.requires_grad == checkpoint_param.requires_grad
        assert (
            torch.isfinite(eager_param).all()
            and torch.isfinite(checkpoint_param).all()
        ), "Non-finite parameter detected"

        if eager_param.requires_grad:
            torch.testing.assert_close(
                checkpoint_param.grad,
                eager_param.grad,
                rtol=1e-5,
                atol=1e-8,
            )


class DummyForwardHook(PhlowerForwardHook):
    def __init__(self):
        self.n_calls = 0

    def __call__(
        self,
        name: str,
        unique_name: str,
        result: IPhlowerTensorCollections,
        **kwards,
    ) -> None:
        self.n_calls += 1


@single_module_cases
def test__checkpoint_keeps_phlower_forward_hook_outside_recomputation(
    file_name: str,
):
    model = _load_model(_DATA_DIR / file_name, checkpoint=True)
    model.train()

    hook = DummyForwardHook()
    model._phlower_modules[0].register_phlower_forward_hook(hook)

    inputs, field_data = _prepare_inputs()
    output = model(inputs, field_data=field_data).unique_item().to_tensor()
    output.sum().backward()

    assert hook.n_calls == 1


@single_module_cases
def test__checkpoint_rejects_core_with_registered_buffer(file_name: str):
    model = _load_model(_DATA_DIR / file_name, checkpoint=True)
    model.train()
    assert len(model._phlower_modules) == 1

    core_module = model._phlower_modules[0].get_core_module()
    core_module.register_buffer("sentinel", torch.ones(1))

    inputs, field_data = _prepare_inputs()

    with pytest.raises(
        PhlowerRunTimeError,
        match="Activation checkpoint does not support modules with buffers",
    ):
        _ = model(inputs, field_data=field_data)
