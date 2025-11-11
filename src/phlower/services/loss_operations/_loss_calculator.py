from __future__ import annotations

import abc
from typing import Literal

import torch

from phlower._base import GraphBatchInfo
from phlower._base._functionals import unbatch
from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.settings._trainer_setting import PhlowerTrainerSetting
from phlower.utils.typing import LossFunctionType


def _mse(pred: PhlowerTensor, answer: PhlowerTensor) -> PhlowerTensor:
    # NOTE: To avoid shape mismatch
    # Ex. (1, N, 3, 1) and (N, 3, 1)
    return torch.nn.functional.mse_loss(
        torch.squeeze(pred.to_tensor()), torch.squeeze(answer.to_tensor())
    )


class PhlowerLossFunctionsFactory:
    _REGISTERED: dict[str, LossFunctionType] = {"mse": _mse}

    @classmethod
    def register(
        cls, name: str, loss_function: LossFunctionType, overwrite: bool = False
    ) -> LossFunctionType:
        if (name not in cls._REGISTERED) or overwrite:
            cls._REGISTERED[name] = loss_function
            return

        raise ValueError(
            f"Loss function named {name} has already existed."
            " If you want to overwrite it, set overwrite=True"
        )

    @classmethod
    def unregister(cls, name: str):
        if name not in cls._REGISTERED:
            raise KeyError(f"{name} does not exist.")

        cls._REGISTERED.pop(name)

    @classmethod
    def get(cls, name: str) -> LossFunctionType:
        if name not in cls._REGISTERED:
            raise KeyError(f"Loss function: {name} is not registered.")
        return cls._REGISTERED[name]


class ILossCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self,
        prediction: IPhlowerTensorCollections,
        answer: IPhlowerTensorCollections,
    ) -> IPhlowerTensorCollections:
        raise NotImplementedError()

    @abc.abstractmethod
    def aggregate(
        self,
        prediction: IPhlowerTensorCollections,
        answer: IPhlowerTensorCollections,
    ) -> PhlowerTensor:
        raise NotImplementedError()


class LossCalculator(ILossCalculator):
    @classmethod
    def from_setting(
        cls,
        setting: PhlowerTrainerSetting,
    ) -> LossCalculator:
        return LossCalculator(**setting.loss_setting.__dict__)

    def __init__(
        self,
        name2loss: dict[str, str],
        name2weight: dict[str, float] | None = None,
        aggregation_method: Literal["sum", "mean"] = "sum",
    ):
        self._name2loss = name2loss
        self._name2weight = name2weight
        self._aggregation_method = aggregation_method
        self._check_loss_function_exists()

    def get_loss_function(self, variable_name: str) -> LossFunctionType:
        if variable_name not in self._name2loss:
            raise ValueError(
                "Loss function is missing. "
                f"Loss function for {variable_name} is not defined."
            )
        loss_name = self._name2loss[variable_name]
        return PhlowerLossFunctionsFactory.get(loss_name)

    def _check_loss_function_exists(self) -> None:
        for loss_name in self._name2loss.values():
            if PhlowerLossFunctionsFactory.get(loss_name) is None:
                raise ValueError(f"Unknown loss function name: {loss_name}")

    def aggregate(self, losses: IPhlowerTensorCollections) -> PhlowerTensor:
        match self._aggregation_method:
            case "sum":
                return losses.sum(weights=self._name2weight)
            case "mean":
                return losses.mean(weights=self._name2weight)
            case _:
                raise ValueError(
                    f"Unknown aggregation method: {self._aggregation_method}. "
                    'Choose from "sum" or "mean".'
                )

    def calculate(
        self,
        prediction: IPhlowerTensorCollections,
        answer: IPhlowerTensorCollections,
        batch_info_dict: dict[str, GraphBatchInfo],
    ) -> IPhlowerTensorCollections:
        loss_items: dict[str, torch.Tensor] = {}
        for key in answer.keys():
            if key not in prediction:
                raise ValueError(
                    f"{key} is not found in predictions. "
                    f"prediction keys: {list(prediction.keys())}"
                )

            batch_info = batch_info_dict[key]
            _preds = unbatch(prediction[key], batch_info)
            _answers = unbatch(answer[key], batch_info)

            loss_func = self.get_loss_function(key)
            loss_items[key] = torch.mean(
                torch.stack(
                    [
                        loss_func(p, a)
                        for p, a in zip(_preds, _answers, strict=True)
                    ]
                )
            )

        return phlower_tensor_collection(loss_items)
