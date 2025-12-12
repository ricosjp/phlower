import pathlib
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from phlower_tensor import (
    SimulationField,
    phlower_array,
    phlower_tensor,
)
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)

from phlower.nn import PhlowerGroupModule
from phlower.nn._group_module import _GroupOptimizeProblem
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


@pytest.mark.parametrize(
    "yaml_file, time_series_length",
    [
        ("forward_time_series.yml", 3),
        ("forward_time_series_any.yml", 7),
        ("forward_time_series_any.yml", 1),
    ],
)
def test__forward_time_series_mode(yaml_file: str, time_series_length: int):
    setting_file = _SAMPLE_SETTING_DIR / yaml_file
    setting = PhlowerSetting.read_yaml(setting_file)

    setting.model.network.resolve(is_first=True)
    group = PhlowerGroupModule.from_setting(setting.model.network)

    input_tensor = phlower_tensor(torch.rand(10, 3))
    time_series_tensor = phlower_tensor(
        torch.rand(time_series_length, 10, 1), is_time_series=True
    )
    phlower_tensors = phlower_tensor_collection(
        {
            "sample_input": input_tensor,
            "time_series_boundary": time_series_tensor,
        }
    )

    results = group.forward(phlower_tensors, field_data=None)
    out = results.unique_item()
    assert out.is_time_series

    assert out.time_series_length == time_series_length


@pytest.mark.parametrize("yaml_file", ["forward_time_series.yml"])
def test__calculation_graph_when_time_series_mode(yaml_file: str):
    setting_file = _SAMPLE_SETTING_DIR / yaml_file
    setting = PhlowerSetting.read_yaml(setting_file)

    setting.model.network.resolve(is_first=True)
    group = PhlowerGroupModule.from_setting(setting.model.network)

    input_tensor = phlower_tensor(torch.rand(2, 3))
    phlower_tensors = phlower_tensor_collection({"sample_input": input_tensor})

    results = group.forward(phlower_tensors, field_data=None)
    out = results.unique_item()
    assert out.is_time_series

    # Check time series tensors are not parent-child
    #  relashionship on calculation graph
    grads = torch.autograd.grad(
        [out[0].to_tensor()],
        [out[1].to_tensor()],
        grad_outputs=[torch.ones_like(out[0].to_tensor())],
        retain_graph=True,
        allow_unused=True,
    )
    assert grads[0] is None


@pytest.mark.parametrize(
    "input_time_series, time_series_length, expected",
    [([5, 0, 0], 5, 5), ([0, 0, 0], 3, 3), ([6, 6, 0], -1, 6)],
)
@mock.patch("phlower.nn._group_module.reduce_stack")
def test__forward_count_when_time_series_input(
    mock_reduce_stack: mock.MagicMock,
    input_time_series: list[int],
    time_series_length: int | None,
    expected: int,
):
    group = PhlowerGroupModule(
        name="test_group",
        modules=[],
        no_grad=False,
        input_keys=["input"],
        destinations=["next_group"],
        output_keys=["output"],
        is_steady_problem=False,
        iteration_solver=None,
        time_series_length=time_series_length,
    )

    mock_input = phlower_tensor_collection(
        {
            f"input_{i}": phlower_tensor(
                torch.rand(t, 10, 1), is_time_series=True
            )
            if t > 0
            else phlower_tensor(torch.rand(10, 1))
            for i, t in enumerate(input_time_series)
        }
    )

    with mock.patch.object(PhlowerGroupModule, "_forward") as mocked:
        group.forward(mock_input, field_data=None)

        assert mocked.call_count == expected


@pytest.mark.parametrize(
    "input_time_series, time_series_length",
    [([5, 0, 0], 5), ([0, 0, 0], 3), ([6, 6, 0], -1)],
)
@mock.patch("phlower.nn._group_module.reduce_stack")
def test__snapshot_input_when_time_series_forward(
    mock_reduce_stack: mock.MagicMock,
    input_time_series: list[int],
    time_series_length: int | None,
):
    group = PhlowerGroupModule(
        name="test_group",
        modules=[],
        no_grad=False,
        input_keys=["input"],
        destinations=["next_group"],
        output_keys=["output"],
        is_steady_problem=False,
        iteration_solver=None,
        time_series_length=time_series_length,
    )

    mock_input = phlower_tensor_collection(
        {
            f"input_{i}": phlower_tensor(
                torch.rand(t, 10, 1), is_time_series=True
            )
            if t > 0
            else phlower_tensor(torch.rand(10, 1))
            for i, t in enumerate(input_time_series)
        }
    )

    with mock.patch.object(PhlowerGroupModule, "_forward") as mocked:
        group.forward(mock_input, field_data=None)

        for args in mocked.call_args_list:
            arg: IPhlowerTensorCollections = args.args[0]
            assert arg.get_time_series_length() == 0


@pytest.mark.parametrize(
    "dict_data_shapes",
    [
        {
            "a": (10, 4),
            "b": (10, 3, 4),
        }
    ],
)
@pytest.mark.parametrize(
    "target_keys",
    [
        ["a"],
        ["b"],
        ["a", "b"],
    ],
)
@pytest.mark.parametrize("steady_mode", [True, False])
def test__subtract_input_for_gradient(
    dict_data_shapes: dict[str, tuple[int]],
    target_keys: list[str],
    steady_mode: bool,
):
    initial = phlower_tensor_collection(
        {k: phlower_tensor(torch.rand(v)) for k, v in dict_data_shapes.items()}
    )
    input_ = phlower_tensor_collection(
        {k: phlower_tensor(torch.rand(v)) for k, v in dict_data_shapes.items()}
    )
    diff = phlower_tensor_collection(
        {k: phlower_tensor(torch.rand(v)) for k, v in dict_data_shapes.items()}
    )

    def step_forward_wo_input(
        input_: IPhlowerTensorCollections,
    ) -> IPhlowerTensorCollections:
        return diff

    def step_forward_w_input(
        input_: IPhlowerTensorCollections,
    ) -> IPhlowerTensorCollections:
        return input_ + diff

    problem_wo_subtract = _GroupOptimizeProblem(
        initials=initial,
        step_forward=step_forward_wo_input,
        steady_mode=steady_mode,
    )
    desired_gradient = problem_wo_subtract.gradient(
        input_, update_keys=target_keys, operator_keys=target_keys
    )

    problem_w_subtract = _GroupOptimizeProblem(
        initials=initial,
        step_forward=step_forward_w_input,
        steady_mode=steady_mode,
    )
    actual_gradient = problem_w_subtract.gradient(
        input_, update_keys=target_keys
    )

    for (actual_key, actual_value), (desired_key, desired_value) in zip(
        actual_gradient.items(), desired_gradient.items(), strict=True
    ):
        assert actual_key == desired_key
        np.testing.assert_almost_equal(
            actual_value.numpy(), desired_value.numpy()
        )
