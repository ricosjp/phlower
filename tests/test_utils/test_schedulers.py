from unittest import mock

import pytest
import torch
from phlower.utils import SchedulerSelector


@pytest.mark.parametrize(
    "name, desired",
    [
        ("LambdaLR", True),
        ("MultiplicativeLR", True),
        ("StepLR", True),
        ("MultiStepLR", True),
        ("ConstantLR", True),
        ("LinearLR", True),
        ("ExponentialLR", True),
        ("SequentialLR", True),
        ("CosineAnnealingLR", True),
        ("ChainedScheduler", True),
        ("ReduceLROnPlateau", True),
        ("CyclicLR", True),
        ("CosineAnnealingWarmRestarts", True),
        ("OneCycleLR", True),
        ("PolynomialLR", True),
        ("MyScheduler", False),
    ],
)
def test__exist(name: str, desired: bool):
    assert SchedulerSelector.exist(name) == desired


@pytest.mark.parametrize(
    "name",
    [
        ("LambdaLR"),
        ("MultiplicativeLR"),
        ("StepLR"),
        ("MultiStepLR"),
        ("ConstantLR"),
        ("LinearLR"),
        ("ExponentialLR"),
        ("SequentialLR"),
        ("CosineAnnealingLR"),
        ("ChainedScheduler"),
        ("ReduceLROnPlateau"),
        ("CyclicLR"),
        ("CosineAnnealingWarmRestarts"),
        ("OneCycleLR"),
        ("PolynomialLR"),
    ],
)
def test__select(name: str):
    scheduler = SchedulerSelector.select(name)
    assert scheduler.__name__ == name


@pytest.mark.parametrize("name", ["MyScheduler", "BestScheduler"])
def test__exist_after_register(name: str):
    assert not SchedulerSelector.exist(name)
    dummy = mock.MagicMock(torch.optim.Optimizer)
    SchedulerSelector.register(name, dummy)
    assert SchedulerSelector.exist(name)

    # Not to affect other tests, unregister scheduler
    SchedulerSelector._REGISTERED.pop(name)
