from __future__ import annotations

import abc
import pathlib
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp
import torch

DenseArrayType = np.ndarray

SparseArrayType = (
    sp.coo_matrix
    | sp.csr_matrix
    | sp.csc_matrix
    | sp.csr_array
    | sp.coo_array
    | sp.csc_array
)

ArrayDataType = np.ndarray | SparseArrayType

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
    output_directory: pathlib.Path | None = None


class AfterEvaluationOutput(NamedTuple):
    """output data after evaluation for one epoch"""

    epoch: int
    train_eval_loss: float
    elapsed_time: float
    output_directory: pathlib.Path | None = None
    validation_eval_loss: float | None = None
    train_loss_details: dict[str, float] | None = None
    validation_loss_details: dict[str, float] | None = None
