from __future__ import annotations

import abc
import pathlib
from collections.abc import Callable, Mapping
from typing import Any, Generic, NamedTuple, TypeVar

import numpy as np
import scipy.sparse as sp
import torch

from phlower.utils.enums import PhlowerHandlerTrigger

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


class AfterEpochTrainingInfo(NamedTuple):
    """output data after training for one epoch"""

    epoch: int
    train_losses: list[float]
    train_loss_details: list[dict[str, float]]
    output_directory: pathlib.Path | None = None


class AfterEvaluationOutput(NamedTuple):
    """output data after evaluation for one epoch"""

    epoch: int
    train_eval_loss: float
    elapsed_time: float | None
    output_directory: pathlib.Path | None = None
    validation_eval_loss: float | None = None
    train_loss_details: dict[str, float] | None = None
    validation_loss_details: dict[str, float] | None = None


T = TypeVar("T", AfterEvaluationOutput, float)


class IPhlowerHandler(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, output: T) -> dict[str, Any]:
        """Run handler's program. If output dictionary conatins "TERMINATE" and
         its value is True, training process is terminated forcefully.

        Args:
            output (AfterEvaluationOutput | float):
                loss value for the current iteration (trigger: iteration)
                output data from evaluation runner (trigger: evaluation)

        Returns:
            dict[str, Any]: output data
        """
        ...

    @abc.abstractmethod
    def state_dict(self) -> dict[str, float]: ...

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Mapping) -> None: ...

    @classmethod
    @abc.abstractmethod
    def trigger(self) -> PhlowerHandlerTrigger: ...
