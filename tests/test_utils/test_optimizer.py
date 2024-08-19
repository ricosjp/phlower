from unittest import mock

import pytest
import torch
from phlower.utils import OptimizerSelector


@pytest.mark.parametrize(
    "name, desired",
    [
        ("Adadelta", True),
        ("Adam", True),
        ("SparseAdam", True),
        ("Adamax", True),
        ("ASGD", True),
        ("LBFGS", True),
        ("NAdam", True),
        ("RAdam", True),
        ("RMSprop", True),
        ("Rprop", True),
        ("SGD", True),
        ("MyOptimizer", False),
    ],
)
def test__exist(name: str, desired: bool):
    assert OptimizerSelector.exist(name) == desired


@pytest.mark.parametrize(
    "name",
    [
        ("Adadelta"),
        ("Adam"),
        ("SparseAdam"),
        ("Adamax"),
        ("ASGD"),
        ("LBFGS"),
        ("NAdam"),
        ("RAdam"),
        ("RMSprop"),
        ("Rprop"),
        ("SGD"),
    ],
)
def test__select(name: str):
    optim = OptimizerSelector.select(name)
    assert optim.__name__ == name


@pytest.mark.parametrize("name", ["MyOptimizer", "BestOptimizer"])
def test__exist_after_register(name: str):
    assert not OptimizerSelector.exist(name)
    dummy = mock.MagicMock(torch.optim.Optimizer)
    OptimizerSelector.register(name, dummy)
    assert OptimizerSelector.exist(name)

    # Not to affect other tests, unregister scheduler
    OptimizerSelector._REGISTERED.pop(name)
