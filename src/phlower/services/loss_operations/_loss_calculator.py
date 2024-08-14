from __future__ import annotations

import abc

import torch

from phlower._base import GraphBatchInfo
from phlower._base._functionals import unbatch
from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower.services.loss_operations._loss_functions import get_loss_function
from phlower.settings._trainer_setting import LossSetting, PhlowerTrainerSetting
from phlower.utils.typing import LossFunctionType


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
        user_loss_functions: dict[str, LossFunctionType] = None,
    ) -> LossCalculator:
        loss_setting = setting.loss_setting
        return cls(
            setting=loss_setting, user_loss_functions=user_loss_functions
        )

    @classmethod
    def from_dict(
        cls,
        name2loss: dict[str, str],
        name2weight: dict[str, float] = None,
        user_loss_functions: dict[str, LossFunctionType] = None,
    ) -> LossCalculator:
        loss_setting = LossSetting(name2loss=name2loss, name2weight=name2weight)

        return cls(
            setting=loss_setting, user_loss_functions=user_loss_functions
        )

    def __init__(
        self,
        setting: LossSetting,
        *,
        user_loss_functions: dict[str, LossFunctionType] = None,
    ):
        self._setting = setting
        if user_loss_functions is None:
            user_loss_functions = {}

        self._user_loss_functions = user_loss_functions

        self._check_loss_function_exists()

    def get_loss_function(self, variable_name: str) -> LossFunctionType:
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

    def aggregate(self, losses: IPhlowerTensorCollections) -> PhlowerTensor:
        if self._setting.name2weight is None:
            return losses.sum()

        return losses.sum(weights=self._setting.name2weight)

    def calculate(
        self,
        prediction: IPhlowerTensorCollections,
        answer: IPhlowerTensorCollections,
        batch_info_dict: dict[str, GraphBatchInfo],
    ) -> IPhlowerTensorCollections:
        loss_items: dict[str, torch.Tensor] = {}
        for key in answer.keys():
            if key not in prediction:
                raise KeyError(
                    f"{key} is not found in predictions. "
                    f"prediction keys: {list(prediction.keys())}"
                )

            batch_info = batch_info_dict[key]
            _preds = unbatch(prediction[key], batch_info)
            _answers = unbatch(answer[key], batch_info)
            if len(_preds) != len(_answers):
                raise ValueError(
                    "Sizes of unbatched tensors in predictions and "
                    "that in answers are not equal."
                )

            loss_func = self.get_loss_function(key)

            # _tmp_losses = [
            #     loss_func(p, a) for p, a in zip(_preds, _answers, strict=True)
            # ]
            # print(f"batch size in loss: {len(_tmp_losses)}")
            # for i in range(len(_tmp_losses)):
            #     print(_tmp_losses[i])

            loss_items[key] = torch.mean(
                torch.stack(
                    [
                        loss_func(p, a)
                        for p, a in zip(_preds, _answers, strict=True)
                    ]
                )
            )

        return phlower_tensor_collection(loss_items)
