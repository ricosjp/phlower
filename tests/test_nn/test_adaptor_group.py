import pathlib

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn import PhlowerGroupModule
from phlower.nn._group_module import PhlowerAdaptorGroupModule
from phlower.settings import PhlowerSetting
from phlower.settings._group_setting import AdaptorGroupModuleSetting

_SAMPLE_SETTING_DIR = pathlib.Path(__file__).parent / "data/adaptor_group"


def identity_group_module_dict() -> dict:
    return {
        "name": "DEMO",
        "no_grad": False,
        "nn_type": "Group",
        "modules": [
            {
                "nn_type": "Identity",
            }
        ],
    }


@given(
    input_keys=st.lists(
        st.text(min_size=1, max_size=5),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x,
    ),
    output_keys=st.lists(
        st.text(min_size=1, max_size=5),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x,
    ),
    destinations=st.lists(
        st.text(min_size=1, max_size=5), min_size=1, max_size=5
    ),
)
def test__apply_setting_items(
    input_keys: list[str], output_keys: list[str], destinations: list[str]
):
    prefixes = ["data_", "residual_"]

    setting = AdaptorGroupModuleSetting(
        **{
            "name": "DEMO",
            "no_grad": False,
            "inputs": [
                {"name": f"{p}{key}", "n_last_dim": 1}
                for key in input_keys
                for p in prefixes
            ],
            "nn_type": "AdaptorGroup",
            "outputs": [
                {
                    "name": f"{p}{key}",
                    "n_last_dim": 1,
                }
                for key in output_keys
                for p in prefixes
            ],
            "destinations": destinations,
            "adaptor_setting": {
                "adaptor_type": "prefix_sorting",
                "prefix_settings": [{"prefix": p} for p in prefixes],
            },
            "network": {
                "name": "DEMO",
                "no_grad": False,
                "nn_type": "Group",
                "modules": [
                    {
                        "name": "DEMO_ID",
                        "nn_type": "Identity",
                    }
                ],
            },
        }
    )
    model = PhlowerAdaptorGroupModule.from_setting(setting)

    assert model._input_keys == [
        f"{p}{key}" for key in input_keys for p in prefixes
    ]
    assert model._output_keys == [
        f"{p}{key}" for key in output_keys for p in prefixes
    ]
    assert model._destinations == destinations
    assert model.name == "DEMO"


def create_preprocessed_data() -> IPhlowerTensorCollections:
    n_node = 30
    return phlower_tensor_collection(
        {
            "data1_feature0": torch.randn(n_node, 10),
            "data2_feature0": torch.randn(n_node, 10),
            "data1_feature1": torch.randn(n_node, 12),
            "data2_feature1": torch.randn(n_node, 12),
        }
    )


@pytest.mark.parametrize(
    "file_name",
    [
        "with_share_nn.yml",
    ],
)
def test__forward_with_prefixes(file_name: str):
    setting = PhlowerSetting.read_yaml(_SAMPLE_SETTING_DIR / file_name)
    setting.model.resolve()

    sample = create_preprocessed_data()
    model = PhlowerGroupModule.from_setting(setting.model.network)

    out = model.forward(sample, field_data=None)

    assert set(out.keys()) == {"data1_out_feature", "data2_out_feature"}
    assert out["data1_out_feature"].shape == (30, 5)
    assert out["data2_out_feature"].shape == (30, 5)
