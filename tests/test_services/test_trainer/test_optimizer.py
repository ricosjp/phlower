from unittest import mock

import pytest
import torch

from phlower.services.trainer._optimizer import PhlowerOptimizerWrapper
from phlower.settings import PhlowerTrainerSetting


@pytest.mark.parametrize(
    "optimizer, optimizer_parameters, schedulers",
    [
        ("Adam", {"lr": 0.003}, []),
        ("SGD", {"lr": 0.1, "weight_decay": 0.01}, []),
        (
            "SGD",
            {"lr": 0.1, "weight_decay": 0.01},
            [
                {
                    "scheduler": "ReduceLROnPlateau",
                    "parameters": {"mode": "min", "patience": 12},
                }
            ],
        ),
    ],
)
def test__pass_kwargs_when_call_from_setting(
    optimizer, optimizer_parameters, schedulers
):
    setting = PhlowerTrainerSetting(
        loss_setting={"name2loss": {}},
        optimizer_setting={
            "optimizer": optimizer,
            "parameters": optimizer_parameters,
        },
        scheduler_setting=schedulers,
    )

    with mock.patch.object(
        PhlowerOptimizerWrapper, "__init__", return_value=None
    ) as mocked:
        model = torch.nn.Linear(in_features=10, out_features=10)
        _ = PhlowerOptimizerWrapper.from_setting(setting, model=model)

        kwargs = mocked.call_args.kwargs
        assert kwargs["optimizer"] == optimizer
        assert kwargs["optimizer_kwargs"] == optimizer_parameters

        for scheduler in schedulers:
            name = scheduler["scheduler"]
            assert name in kwargs["schedulers"]
            assert kwargs["schedulers"][name] == scheduler["parameters"]


@pytest.mark.parametrize(
    "optimizer, lr, weight_decay, desired_optimizer",
    [
        ("Adam", 0.001, 0, torch.optim.Adam),
        ("Adam", 0.0003, 0, torch.optim.Adam),
        ("SGD", 0.0005, 0.01, torch.optim.SGD),
    ],
)
def test__optimizer_parameters(optimizer, lr, weight_decay, desired_optimizer):
    model = torch.nn.Linear(in_features=10, out_features=10)
    optimizer = PhlowerOptimizerWrapper(
        parameters=model.parameters(),
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr, "weight_decay": weight_decay},
        schedulers={},
    )

    assert isinstance(optimizer._optimizer, desired_optimizer)

    state_dict = optimizer.state_dict()
    assert len(state_dict["param_groups"]) == 1
    assert state_dict["param_groups"][0]["lr"] == lr
    assert state_dict["param_groups"][0]["weight_decay"] == weight_decay


@pytest.mark.parametrize(
    "schedulers, desired",
    [
        (
            {
                "ReduceLROnPlateau": {"mode": "min", "patience": 21},
                "StepLR": {"step_size": 30, "gamma": 0.2},
            },
            [
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                torch.optim.lr_scheduler.StepLR,
            ],
        ),
        (
            {
                "CosineAnnealingLR": {"T_max": 10},
                "ConstantLR": {"factor": 0.5},
            },
            [
                torch.optim.lr_scheduler.CosineAnnealingLR,
                torch.optim.lr_scheduler.ConstantLR,
            ],
        ),
    ],
)
def test__scheduler_parameters(schedulers, desired):
    dummy = torch.nn.Linear(in_features=10, out_features=10)
    optimizer = PhlowerOptimizerWrapper(
        parameters=dummy.parameters(),
        optimizer="Adam",
        optimizer_kwargs={},
        schedulers=schedulers,
    )

    assert len(optimizer._schedulers) == len(schedulers)
    for i, _scheduler in enumerate(desired):
        isinstance(optimizer._schedulers[i], _scheduler)

    for i, _scheduler in enumerate(desired):
        params = schedulers[_scheduler.__name__]
        for k, v in params.items():
            assert getattr(optimizer._schedulers[i], k) == v
