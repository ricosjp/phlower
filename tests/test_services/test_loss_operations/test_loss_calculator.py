import pathlib

import numpy as np
import pytest
import torch
from phlower import PhlowerTensor, phlower_tensor
from phlower._base import GraphBatchInfo
from phlower._base._functionals import to_batch
from phlower.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.services.loss_operations import (
    LossCalculator,
    PhlowerLossFunctionsFactory,
)
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
    "losses, name2weight, desired",
    [
        ({"u": 1.0, "p": 1.5}, None, 2.5),
        ({"u": 1.0, "p": 1.5}, {"u": 1.0, "p": 0.0}, 1.0),
        ({"u": 1.0, "p": 1.5}, {"u": 1.0, "p": 2.0}, 4.0),
    ],
)
def test__aggregate(
    losses: dict[str, float],
    name2weight: dict[str, float] | None,
    desired: float,
):
    losses = phlower_tensor_collection(
        {k: torch.tensor(v) for k, v in losses.items()}
    )
    calculator = LossCalculator(
        name2loss={"u": "mse", "p": "mse"}, name2weight=name2weight
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

    with pytest.raises(ValueError) as ex:
        _ = calculator.calculate(predictions, answers, batch_info)
    assert "Loss function is missing" in str(ex.value)


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
