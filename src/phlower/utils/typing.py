from __future__ import annotations

import abc
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp
import torch

ArrayDataType = np.ndarray | sp.coo_matrix | sp.csr_matrix | sp.csc_matrix

DenseArrayType = np.ndarray

SparseArrayType = sp.coo_matrix | sp.csr_matrix | sp.csc_matrix

LossFunctionType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class PhlowerHandlerType(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, output: AfterEvaluationOutput) -> dict[str, Any]:
        """Run handler's program. If output dictionary conatins "TERMINATE" and
         its value is True, training process is terminated forcefully.

        Args:
            output (AfterEvaluationOutput): output data from evaluation runner

        Returns:
            dict[str, Any]: output data
        """
        ...

    @abc.abstractmethod
    def state_dict(self) -> dict[str, float]: ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Mapping) -> None: ...


class AfterEpochTrainingInfo(NamedTuple):
    """output data after training for one epoch"""

    epoch: int
    train_losses: list[float]


class AfterEvaluationOutput(NamedTuple):
    """output data after evaluation for one epoch"""

    train_eval_loss: float
    validation_eval_loss: float | None
