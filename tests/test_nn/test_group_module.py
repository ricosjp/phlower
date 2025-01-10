import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from phlower import phlower_tensor
from phlower._base import phlower_array
from phlower._fields import SimulationField
from phlower.collections import phlower_tensor_collection
from phlower.nn import PhlowerGroupModule
from phlower.settings import PhlowerSetting

_SAMPLE_SETTING_DIR = pathlib.Path(__file__).parent / "data/group"


@pytest.mark.parametrize("yaml_file", ["with_share_nn.yml"])
def test__resolve_modules_from_setting(yaml_file: str):
    setting_file = _SAMPLE_SETTING_DIR / yaml_file
    setting = PhlowerSetting.read_yaml(setting_file)

    setting.model.network.resolve(is_first=True)
    _ = PhlowerGroupModule.from_setting(setting.model.network)


@pytest.mark.parametrize(
    "yaml_file, coeff", [("forward_test.yml", 1.0), ("coeff_test.yml", -1.0)]
)
def test__can_load_coeff_for_module(
    yaml_file: str,
    coeff: float,
):
    setting_file = _SAMPLE_SETTING_DIR / yaml_file
    setting = PhlowerSetting.read_yaml(setting_file)

    setting.model.network.resolve(is_first=True)
    adapter = PhlowerGroupModule.from_setting(setting.model.network)

    input_tensor = phlower_tensor(torch.rand(2, 3))
    phlower_tensors = phlower_tensor_collection({"sample_input": input_tensor})

    actual = adapter.forward(phlower_tensors, field_data=None)
    np.testing.assert_almost_equal(
        actual["sample_output"].to_numpy(), coeff * input_tensor.to_numpy()
    )


@pytest.mark.parametrize("yaml_file", ["with_share_nn.yml"])
def test__draw(yaml_file: str):
    output_directory = pathlib.Path(__file__).parent / "out"

    setting_file = _SAMPLE_SETTING_DIR / yaml_file
    setting = PhlowerSetting.read_yaml(setting_file)

    setting.model.network.resolve(is_first=True)
    group = PhlowerGroupModule.from_setting(setting.model.network)

    group.draw(output_directory)


@pytest.mark.parametrize(
    "yaml_file, input_n_feature, n_nodes", [("with_share_nn.yml", 10, 20)]
)
def test__forward_and_backward(
    yaml_file: str, input_n_feature: int, n_nodes: int
):
    setting_file = _SAMPLE_SETTING_DIR / yaml_file
    setting = PhlowerSetting.read_yaml(setting_file)

    setting.model.network.resolve(is_first=True)
    group = PhlowerGroupModule.from_setting(setting.model.network)

    phlower_tensors = phlower_tensor_collection(
        {
            "feature0": phlower_array(
                np.random.rand(n_nodes, input_n_feature).astype(np.float32)
            ).to_tensor(),
            "feature1": phlower_array(
                np.random.rand(n_nodes, input_n_feature).astype(np.float32)
            ).to_tensor(),
        }
    )

    rng = np.random.default_rng()
    sparse_adj = phlower_array(
        sp.random(
            n_nodes, n_nodes, density=0.1, random_state=rng, dtype=np.float32
        )
    )
    nodal_nadj = phlower_tensor(sparse_adj.to_tensor())
    field_data = SimulationField(field_tensors={"support1": nodal_nadj})

    output = group.forward(data=phlower_tensors, field_data=field_data)

    actual = output.unique_item()
    dummy_label = torch.rand(*actual.shape)
    loss = torch.nn.functional.mse_loss(actual, dummy_label)

    loss.backward()
