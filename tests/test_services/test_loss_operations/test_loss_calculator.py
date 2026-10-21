import pathlib

import numpy as np
import pytest
import torch
from phlower_tensor import (
    GraphBatchInfo,
    PhlowerTensor,
    SimulationField,
    phlower_tensor,
)
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower_tensor.functionals import to_batch, unbatch

from phlower.services.loss_operations import (
    LossCalculator,
    PhlowerLossFunctionsFactory,
)
from phlower.services.loss_operations._loss_calculator import _select_nodes
from phlower.settings import PhlowerSetting

DATA_DIR = pathlib.Path(__file__).parent / "data"


@pytest.mark.parametrize("name", ["loss_not_defined", "aaaaaa"])
def test__not_unregister_unexisted_loss(name: str):
    with pytest.raises(KeyError) as ex:
        PhlowerLossFunctionsFactory.unregister(name)

    assert f"{name} does not exist." in str(ex.value)


@pytest.fixture
def setup_user_loss_function():
    def user_loss_sse(pred: torch.Tensor, ans: torch.Tensor) -> torch.Tensor:
        return torch.sum((pred - ans) ** 2.0)

    PhlowerLossFunctionsFactory.register("user_sse", user_loss_sse)

    yield None
    PhlowerLossFunctionsFactory.unregister("user_sse")


def test__not_allowed_overwrite(setup_user_loss_function: None):
    def dummy_loss(pred: torch.Tensor, ans: torch.Tensor) -> torch.Tensor:
        return pred

    with pytest.raises(ValueError) as ex:
        PhlowerLossFunctionsFactory.register("user_sse", dummy_loss)
    assert "Loss function named user_sse has already existed." in str(ex.value)


@pytest.mark.parametrize(
    "yaml_file_name, variable_name",
    [
        ("base_loss_function.yml", "nodal_last_u"),
        ("user_loss_function.yml", "nodal_last_u"),
    ],
)
def test__get_loss_function(
    yaml_file_name: str, variable_name: str, setup_user_loss_function: None
):
    setting = PhlowerSetting.read_yaml(DATA_DIR / yaml_file_name)
    calculator = LossCalculator.from_setting(setting.training)

    assert calculator.get_loss_function(variable_name) is not None


@pytest.mark.parametrize(
    "losses, name2weight, aggregation_method, desired",
    [
        ({"u": 1.0, "p": 1.5}, None, "sum", 2.5),
        ({"u": 1.0, "p": 1.5}, {"u": 1.0, "p": 0.0}, "sum", 1.0),
        ({"u": 1.0, "p": 1.5}, {"u": 1.0, "p": 2.0}, "sum", 4.0),
        ({"u": 1.0, "p": 1.5}, None, "mean", 1.25),
        ({"u": 1.0, "p": 1.5}, {"u": 1.0, "p": 0.0}, "mean", 1.0),
        ({"u": 1.0, "p": 1.5}, {"u": 1.0, "p": 2.0}, "mean", 4.0 / 3.0),
    ],
)
def test__aggregate(
    losses: dict[str, float],
    name2weight: dict[str, float] | None,
    aggregation_method: str,
    desired: float,
):
    losses = phlower_tensor_collection(
        {k: torch.tensor(v) for k, v in losses.items()}
    )
    calculator = LossCalculator(
        name2loss={"u": "mse", "p": "mse"},
        name2weight=name2weight,
        aggregation_method=aggregation_method,
    )
    actual = calculator.aggregate(losses)

    np.testing.assert_almost_equal(desired, actual.numpy().item())


def create_random_dataset(
    name2shape: dict[str, list[tuple[int]]],
    overwrite_answer_shape: dict[str, list[tuple[int]]] | None = None,
) -> tuple[
    IPhlowerTensorCollections,
    IPhlowerTensorCollections,
    dict[str, GraphBatchInfo],
]:
    answer_shape = overwrite_answer_shape or name2shape

    prediction: dict[str, PhlowerTensor] = {}
    answer: dict[str, PhlowerTensor] = {}
    batch_info: dict[str, GraphBatchInfo] = {}

    for name, shapes in name2shape.items():
        preds = [phlower_tensor(np.random.rand(*shape)) for shape in shapes]
        ans = [
            phlower_tensor(np.random.rand(*shape))
            for shape in answer_shape[name]
        ]

        preds, info = to_batch(preds)
        ans, _ = to_batch(ans)

        prediction[name] = preds
        answer[name] = ans
        batch_info[name] = info
    return (
        phlower_tensor_collection(prediction),
        phlower_tensor_collection(answer),
        batch_info,
    )


@pytest.mark.parametrize(
    "name2loss, name2shape",
    [
        (
            {"nodal_u": "mse"},
            {"nodal_u": [(10, 3, 1), (13, 3, 1)]},
        ),
        (
            {"nodal_u": "mse", "nodal_p": "user_sse"},
            {
                "nodal_u": [(10, 3, 1), (13, 3, 1)],
                "nodal_p": [(10, 1), (13, 1)],
            },
        ),
    ],
)
def test__calculate_batch_data(
    name2loss: dict[str, str],
    name2shape: dict[str, list[tuple[int]]],
    setup_user_loss_function: None,
):
    calculator = LossCalculator(name2loss=name2loss)

    predictions, answers, batch_info = create_random_dataset(name2shape)
    actual = calculator.calculate(predictions, answers, batch_info)

    for k in name2shape:
        assert k in actual
        assert actual[k].size().numel() <= 1


@pytest.mark.parametrize(
    "name2loss, name2shape, answer_shape",
    [
        (
            {"nodal_u": "mse"},
            {"nodal_u": [(10, 3, 1), (13, 3, 1)]},
            {"nodal_u": [(10, 3, 1), (13, 3, 1), (10, 3, 1)]},
        ),
        (
            {"nodal_u": "mse", "nodal_p": "user_sse"},
            {
                "nodal_u": [(10, 3, 1), (13, 3, 1)],
                "nodal_p": [(10, 1), (13, 1)],
            },
            {
                "nodal_u": [(13, 3, 1)],
                "nodal_p": [(10, 1)],
            },
        ),
    ],
)
def test__raise_error_uncompatible_batch_length(
    name2loss: dict[str, str],
    name2shape: dict[str, list[tuple[int]]],
    answer_shape: dict[str, list[tuple[int]]],
    setup_user_loss_function: None,
):
    calculator = LossCalculator(name2loss=name2loss)

    predictions, answers, batch_info = create_random_dataset(
        name2shape, overwrite_answer_shape=answer_shape
    )

    with pytest.raises(RuntimeError):
        _ = calculator.calculate(predictions, answers, batch_info)


@pytest.mark.parametrize(
    "name2loss, name2shape",
    [
        (
            {"nodal_u": "mse"},
            {
                "nodal_u": [(10, 3, 1), (13, 3, 1)],
                "nodal_p": [(10, 1), (13, 1)],
            },
        ),
        (
            {"nodal_p": "mse"},
            {
                "nodal_u": [(10, 3, 1), (13, 3, 1)],
                "nodal_p": [(10, 1), (13, 1)],
            },
        ),
    ],
)
def test__raise_error_when_key_is_missing(
    name2loss: dict[str, str],
    name2shape: dict[str, list[tuple[int]]],
    setup_user_loss_function: None,
):
    calculator = LossCalculator(name2loss=name2loss)

    predictions, answers, batch_info = create_random_dataset(name2shape)

    with pytest.raises(ValueError, match="not found in loss function settings"):
        _ = calculator.calculate(predictions, answers, batch_info)


@pytest.mark.parametrize(
    "name2loss, name2shape, missing_key",
    [
        (
            {"nodal_u": "mse", "nodal_p": "mse"},
            {
                "nodal_u": [(10, 3, 1), (13, 3, 1)],
                "nodal_p": [(10, 1), (13, 1)],
            },
            "nodal_u",
        ),
        (
            {"nodal_p": "mse", "nodal_u": "user_sse"},
            {
                "nodal_u": [(10, 3, 1), (13, 3, 1)],
                "nodal_p": [(10, 1), (13, 1)],
            },
            "nodal_p",
        ),
    ],
)
def test__raise_error_when_prediction_and_answer_not_match(
    name2loss: dict[str, str],
    name2shape: dict[str, list[tuple[int]]],
    missing_key: str,
    setup_user_loss_function: None,
):
    calculator = LossCalculator(name2loss=name2loss)

    predictions, answers, batch_info = create_random_dataset(name2shape)
    predictions.pop(missing_key)

    with pytest.raises(ValueError) as ex:
        _ = calculator.calculate(predictions, answers, batch_info)
    assert f"{missing_key} is not found in predictions" in str(ex.value)


@pytest.mark.parametrize(
    "name2loss, name2shape, answer_shape",
    [
        (
            {"nodal_u": "mse"},
            {"nodal_u": [(10, 10, 10, 3, 1)]},
            {"nodal_u": [(10, 10, 10, 3, 1)]},
        ),
        (
            {"nodal_u": "mse", "nodal_p": "user_sse"},
            {
                "nodal_u": [(4, 4, 4, 3, 1)],
                "nodal_p": [(4, 4, 4, 1)],
            },
            {
                "nodal_u": [(4, 4, 4, 3, 1)],
                "nodal_p": [(4, 4, 4, 1)],
            },
        ),
    ],
)
def test__compute_loss_for_not_batched_voxel_tensor(
    name2loss: dict[str, str],
    name2shape: dict[str, list[tuple[int, ...]]],
    answer_shape: dict[str, list[tuple[int, ...]]],
    setup_user_loss_function: None,
):
    predictions: dict[str, PhlowerTensor] = {}
    answers: dict[str, PhlowerTensor] = {}
    batch_info: dict[str, GraphBatchInfo] = {}

    for name, shapes in name2shape.items():
        preds = [phlower_tensor(np.random.rand(*shape)) for shape in shapes]
        ans = [
            phlower_tensor(np.random.rand(*shape), is_voxel=True)
            for shape in answer_shape[name]
        ]

        preds, info = to_batch(preds)
        ans, _ = to_batch(ans)

        predictions[name] = preds
        answers[name] = ans
        batch_info[name] = info

    calculator = LossCalculator(name2loss=name2loss)
    _ = calculator.calculate(predictions, answers, batch_info)


@pytest.mark.parametrize(
    "name2loss, name2shape",
    [
        (
            {"nodal_residual": "residual_mse"},
            {"nodal_residual": [(10, 3, 1), (13, 3, 1)]},
        ),
    ],
)
def test__compute_residual_losses(
    name2loss: dict[str, str],
    name2shape: dict[str, list[tuple[int, ...]]],
):

    predictions, _, batch_info = create_random_dataset(name2shape)

    answers = phlower_tensor_collection({})

    calculator = LossCalculator(name2loss=name2loss)
    actual = calculator.calculate(predictions, answers, batch_info)

    assert "nodal_residual" in actual

    preds = unbatch(predictions["nodal_residual"], batch_info["nodal_residual"])
    expected = torch.mean(
        torch.stack([torch.mean(p**2.0) for p in preds], dim=0)
    )
    assert torch.allclose(
        actual["nodal_residual"].to_tensor(), expected.to_tensor()
    )


def test__compute_losses_for_value_which_does_not_have_batch_info():

    # when computing residual loss, batch info is not required.
    name2loss = {"nodal_residual": "residual_mse"}
    name2shape = {"nodal_residual": [(10, 3, 1)]}

    predictions, _, _ = create_random_dataset(name2shape)

    # Mimic the case where answer is not provided
    answers = phlower_tensor_collection({})

    calculator = LossCalculator(name2loss=name2loss)
    actual = calculator.calculate(predictions, answers, {})

    assert "nodal_residual" in actual

    expected = torch.mean(predictions["nodal_residual"] ** 2.0)
    assert torch.allclose(
        actual["nodal_residual"].to_tensor(), expected.to_tensor()
    )


def test__loss_names_and_weights_from_losses():
    calculator = LossCalculator(
        losses=[
            {
                "name": "loss_u",
                "variable": "nodal_u",
                "loss_function": "mse",
                "weight": 2.0,
            },
            {"variable": "nodal_p", "loss_function": "mse"},
        ]
    )
    predictions, answers, batch_info = create_random_dataset(
        {"nodal_u": [(10, 3, 1)], "nodal_p": [(10, 1)]}
    )

    losses = calculator.calculate(predictions, answers, batch_info)
    assert set(losses.keys()) == {"loss_u", "nodal_p"}

    desired = 2.0 * losses["loss_u"].to_tensor() + losses["nodal_p"].to_tensor()
    np.testing.assert_almost_equal(
        calculator.aggregate(losses).numpy().item(), desired.item()
    )


def _masked_loss_calculator() -> LossCalculator:
    return LossCalculator(
        losses=[
            {"variable": "nodal_u", "loss_function": "mse", "mask": "loss_mask"}
        ]
    )


def _create_mask_field(masks: list[np.ndarray]) -> SimulationField:
    batched_masks, batch_info = to_batch([phlower_tensor(m) for m in masks])
    return SimulationField(
        field_tensors=phlower_tensor_collection({"loss_mask": batched_masks}),
        batch_info={"loss_mask": batch_info},
    )


@pytest.mark.parametrize(
    "n_nodes, masked_nodes, is_time_series",
    [
        ([10, 13], [[0, 3, 4], [1, 12]], False),
        ([10], [[2, 5, 6, 7]], True),
    ],
)
def test__calculate_masked_loss(
    n_nodes: list[int], masked_nodes: list[list[int]], is_time_series: bool
):
    shape = (3, 1)
    preds, answers, masks, desired = [], [], [], []
    for n, nodes in zip(n_nodes, masked_nodes, strict=True):
        t_shape = (1, n, *shape) if is_time_series else (n, *shape)
        p = np.random.randn(*t_shape)
        a = np.random.randn(*t_shape)
        m = np.zeros((n, 1))
        m[nodes] = 1.0
        preds.append(phlower_tensor(p, is_time_series=is_time_series))
        answers.append(phlower_tensor(a, is_time_series=is_time_series))
        masks.append(m)
        diff = (p - a)[:, nodes] if is_time_series else (p - a)[nodes]
        desired.append(np.mean(diff**2))

    preds, info = to_batch(preds)
    answers, _ = to_batch(answers)

    actual = _masked_loss_calculator().calculate(
        phlower_tensor_collection({"nodal_u": preds}),
        phlower_tensor_collection({"nodal_u": answers}),
        {"nodal_u": info},
        field_data=_create_mask_field(masks),
    )

    np.testing.assert_almost_equal(
        actual["nodal_u"].numpy().item(), np.mean(desired)
    )


def test__calculate_masked_residual_loss():
    n_nodes = [10, 13]
    masked_nodes = [[0, 3, 4], [1, 12]]
    preds, masks, desired = [], [], []
    for n, nodes in zip(n_nodes, masked_nodes, strict=True):
        p = np.random.randn(n, 3, 1)
        m = np.zeros((n, 1))
        m[nodes] = 1.0
        preds.append(phlower_tensor(p))
        masks.append(m)
        desired.append(np.mean(p[nodes] ** 2))

    preds, _ = to_batch(preds)
    calculator = LossCalculator(
        losses=[
            {
                "variable": "nodal_r",
                "loss_function": "residual_mse",
                "mask": "loss_mask",
            }
        ]
    )

    actual = calculator.calculate(
        phlower_tensor_collection({"nodal_r": preds}),
        phlower_tensor_collection({}),
        {},
        field_data=_create_mask_field(masks),
    )

    np.testing.assert_almost_equal(
        actual["nodal_r"].numpy().item(), np.mean(desired)
    )


def test__raise_error_when_mask_field_is_missing():
    predictions, answers, batch_info = create_random_dataset(
        {"nodal_u": [(10, 3, 1)]}
    )

    with pytest.raises(ValueError, match="Mask field loss_mask"):
        _masked_loss_calculator().calculate(predictions, answers, batch_info)


@pytest.mark.parametrize(
    "mask, desired_msg",
    [
        (np.zeros((10, 1)), "No node is selected"),
        (np.ones((9, 1)), "does not match"),
        (np.ones((10, 2)), "does not match"),
        # A 0/1 mask after standard scaling
        (np.repeat([[-1.0], [1.0]], 5, axis=0), "must be 0 or 1"),
    ],
)
def test__raise_error_for_invalid_mask(mask: np.ndarray, desired_msg: str):
    predictions, answers, batch_info = create_random_dataset(
        {"nodal_u": [(10, 3)]}
    )

    with pytest.raises(ValueError, match=desired_msg):
        _masked_loss_calculator().calculate(
            predictions,
            answers,
            batch_info,
            field_data=_create_mask_field([mask]),
        )


def test__raise_error_for_mask_on_voxel_data():
    voxel = phlower_tensor(np.random.rand(2, 2, 2, 3), is_voxel=True)
    mask = phlower_tensor(np.ones((8, 1)))

    with pytest.raises(ValueError, match="not supported for voxel"):
        _select_nodes(voxel, mask)


def test__raise_error_for_time_series_mask():
    tensor = phlower_tensor(np.random.rand(10, 3))
    mask = phlower_tensor(np.ones((5, 10, 1)), is_time_series=True)

    with pytest.raises(ValueError, match="does not match"):
        _select_nodes(tensor, mask)


@pytest.mark.parametrize("is_time_series", [False, True])
def test__select_nodes_keeps_dimension(is_time_series: bool):
    value = np.random.rand(*((1, 5, 3) if is_time_series else (5, 3)))
    tensor = phlower_tensor(
        value, dimension={"L": 1, "T": -1}, is_time_series=is_time_series
    )
    mask = phlower_tensor(np.array([[1.0], [0.0], [1.0], [0.0], [0.0]]))

    actual = _select_nodes(tensor, mask)

    desired = value[:, [0, 2]] if is_time_series else value[[0, 2]]
    np.testing.assert_array_almost_equal(actual.numpy(), desired)
    assert actual.dimension == tensor.dimension
    assert actual.is_time_series == is_time_series
