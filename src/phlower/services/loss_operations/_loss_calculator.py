from __future__ import annotations

import abc
from typing import Callable, Optional, Union

import numpy as np
import torch
from torch.nn import functional
from torch import Tensor

from phlower.base.tensors import PhlowerTensor
from phlower.utils.typing import LossFunctionType
from phlower.settings._phlower_setting import LossSettings
from phlower.collections.tensors import IPhlowerTensorCollections, phlower_tensor_collection
from ._loss_functions import get_loss_function


class ILossCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self,
        prediction: IPhlowerTensorCollections,
        answer: IPhlowerTensorCollections
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
    def from_dict(
        cls,
        name2loss: dict[str, str],
        name2weight: dict[str, float] = None,
        user_loss_functions: dict[str, LossFunctionType] = None,
    ) -> LossCalculator:
        loss_setting = LossSettings(name2loss=name2loss, name2weight=name2weight)

        return cls(setting=loss_setting, user_loss_functions=user_loss_functions)

    def __init__(
        self,
        setting: LossSettings,
        *,
        user_loss_functions: dict[str, LossFunctionType] = None,
    ):
        self._setting = setting
        if user_loss_functions is None:
            user_loss_functions = {}

        self._user_loss_functions = user_loss_functions

        self._check_loss_function_exists()

    def get_loss_function(
        self,
        variable_name: str
    ) -> LossFunctionType:
        loss_name = self._setting.name2loss[variable_name]

        # prioritize user loss functions
        if func := self._user_loss_functions.get(loss_name):
            return func

        return get_loss_function(loss_name)

    def _check_loss_function_exists(self) -> None:
        for loss_name in self._setting.loss_names():
            if self._user_loss_functions.get(loss_name) is not None:
                continue

            if get_loss_function(loss_name) is None:
                raise ValueError(f"Unknown loss function name: {loss_name}")

    def aggregate(
        self,
        losses: IPhlowerTensorCollections    
    ) -> PhlowerTensor:
        if self._setting.name2weight is None:
            return losses.sum()

        return losses.sum(weights=self._setting.name2weight)

    def calculate(
        self,
        prediction: IPhlowerTensorCollections,
        answer: IPhlowerTensorCollections
    ) -> IPhlowerTensorCollections:

        loss_items: dict[str, torch.Tensor] = {}
        for key in answer.keys():
            if key not in prediction:
                raise KeyError(
                    f"{key} is not found in predictions. "
                    f"prediction keys: {list(prediction.keys())}"
                )

            _preds = prediction[key].split()
            _ans = answer[key].split()
            if len(_preds) != len(_preds):
                raise ValueError("Size of splited tensors in predictions and answers is not equal.")

            loss_func = self.get_loss_function(key)
            loss_items[key] = torch.mean(
                torch.stack([loss_func(p, a) for p, a in zip(_preds, _ans)])
            )

        return phlower_tensor_collection(loss_items)        
