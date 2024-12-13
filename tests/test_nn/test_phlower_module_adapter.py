import pathlib

import numpy as np
import pytest
import torch
from phlower import PhlowerTensor
from phlower.collections import phlower_tensor_collection
from phlower.nn import PhlowerGroupModule
from phlower.settings import PhlowerSetting

_SAMPLE_SETTING_DIR = pathlib.Path("tests/test_nn/data")


@pytest.mark.parametrize(
    "yaml_file, coeff", [("forward_test.yml", 1.0), ("coeff_test.yml", -1.0)]
)
def test_forward(
    yaml_file: str,
    coeff: float,
):
    setting_file = _SAMPLE_SETTING_DIR / yaml_file
    setting = PhlowerSetting.read_yaml(setting_file)

    setting.model.network.resolve(is_first=True)
    adapter = PhlowerGroupModule.from_setting(setting.model.network)

    phlower_tensor = PhlowerTensor(torch.rand(2, 3))
    phlower_tensors = phlower_tensor_collection(
        {"sample_input": phlower_tensor}
    )

    actual = adapter.forward(phlower_tensors, field_data=None)
    np.testing.assert_almost_equal(
        actual["sample_output"].to_numpy(), coeff * phlower_tensor.to_numpy()
    )
