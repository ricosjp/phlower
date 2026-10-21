from __future__ import annotations

import abc
from typing import Literal

import torch
from phlower_tensor import GraphBatchInfo, ISimulationField, PhlowerTensor
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from phlower_tensor.functionals import unbatch

from phlower.settings._trainer_setting import (
    EachLossSetting,
    LossSetting,
    PhlowerTrainerSetting,
)
from phlower.utils import get_logger
from phlower.utils.typing import LossFunctionType

_logger = get_logger(__name__)


class ILossFunctionHelper(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def need_answer(self) -> bool: ...

    @abc.abstractmethod
    def __call__(
        self, pred: PhlowerTensor, answer: PhlowerTensor | None
    ) -> PhlowerTensor: ...


class _MSEHelper(ILossFunctionHelper):
    def need_answer(self) -> bool:
        return True

    def __call__(
        self, pred: PhlowerTensor, answer: PhlowerTensor
    ) -> PhlowerTensor:
        # NOTE: To avoid shape mismatch
        # Ex. (1, N, 3, 1) and (N, 3, 1)
        return torch.nn.functional.mse_loss(
            torch.squeeze(pred.to_tensor()), torch.squeeze(answer.to_tensor())
        )


class _UserFunctionHelper(ILossFunctionHelper):
    def __init__(self, loss_function: LossFunctionType):
        self._loss_function = loss_function

    def need_answer(self) -> bool:
        return True

    def __call__(
        self, pred: PhlowerTensor, answer: PhlowerTensor
    ) -> PhlowerTensor:
        return self._loss_function(pred, answer)


class _ResidualMSEHelper(ILossFunctionHelper):
    def need_answer(self) -> bool:
        return False

    def __call__(
        self, pred: PhlowerTensor, answer: Literal[None]
    ) -> PhlowerTensor:
        if answer is not None:
            raise ValueError(
                "Residual loss function does not require answer tensor."
            )
        return torch.mean(torch.square(pred.to_tensor()))


class PhlowerLossFunctionsFactory:
    _REGISTERED: dict[str, ILossFunctionHelper] = {
        "mse": _MSEHelper(),
        "residual_mse": _ResidualMSEHelper(),
    }

    @classmethod
    def register(
        cls, name: str, loss_function: LossFunctionType, overwrite: bool = False
    ) -> LossFunctionType:
        if (name not in cls._REGISTERED) or overwrite:
            cls._REGISTERED[name] = _UserFunctionHelper(loss_function)
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
    def get(cls, name: str) -> ILossFunctionHelper:
        if name not in cls._REGISTERED:
            raise KeyError(f"Loss function: {name} is not registered.")
        return cls._REGISTERED[name]


class ILossCalculator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def calculate(
        self,
        prediction: IPhlowerTensorCollections,
        answer: IPhlowerTensorCollections,
        batch_info_dict: dict[str, GraphBatchInfo],
        field_data: ISimulationField | None = None,
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
        return LossCalculator(
            losses=setting.loss_setting.losses,
            aggregation_method=setting.loss_setting.aggregation_method,
        )

    def __init__(
        self,
        losses: list[EachLossSetting | dict] | None = None,
        name2loss: dict[str, str] | None = None,
        name2weight: dict[str, float] | None = None,
        aggregation_method: Literal["sum", "mean"] = "sum",
    ):
        setting = LossSetting(
            losses=losses or [],
            name2loss=name2loss,
            name2weight=name2weight,
            aggregation_method=aggregation_method,
        )
        self._variable2loss = {v.variable: v for v in setting.losses}
        self._name2weight = {v.name: v.weight for v in setting.losses}
        self._aggregation_method = aggregation_method
        self._check_loss_function_exists()

    def get_loss_function(self, variable_name: str) -> LossFunctionType:
        if variable_name not in self._variable2loss:
            raise ValueError(
                "Loss function is missing. "
                f"Loss function for {variable_name} is not defined."
            )
        loss_name = self._variable2loss[variable_name].loss_function
        return PhlowerLossFunctionsFactory.get(loss_name)

    def _check_loss_function_exists(self) -> None:
        for loss in self._variable2loss.values():
            if PhlowerLossFunctionsFactory.get(loss.loss_function) is None:
                raise ValueError(
                    f"Unknown loss function name: {loss.loss_function}"
                )

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
        field_data: ISimulationField | None = None,
    ) -> IPhlowerTensorCollections:
        loss_items: dict[str, torch.Tensor] = {}

        if not set(answer.keys()).issubset(set(self._variable2loss.keys())):
            raise ValueError(
                f"Some answer keys {list(answer.keys())} "
                "are not found in loss function settings "
                f"{list(self._variable2loss.keys())}."
            )

        for key, loss_setting in self._variable2loss.items():
            if key not in prediction.keys():
                raise ValueError(
                    f"{key} is not found in predictions. "
                    f"prediction keys: {list(prediction.keys())}, "
                    f" answer keys: {list(answer.keys())}"
                )

            loss_func = self.get_loss_function(key)
            if loss_func.need_answer() and key not in answer.keys():
                raise ValueError(
                    f"{key} is not found in answers. "
                    f"prediction keys: {list(prediction.keys())}, "
                    f" answer keys: {list(answer.keys())}"
                )

            if loss_setting.mask is not None:
                _masks = _unbatch_mask(loss_setting.mask, field_data)
                # NOTE: The mask gives the batch info of a prediction
                # without answer.
                batch_info = batch_info_dict.get(
                    key, field_data.get_batch_info(loss_setting.mask)
                )
            else:
                _masks = None
                batch_info = batch_info_dict.get(key)

            if batch_info is not None:
                _preds = unbatch(prediction[key], batch_info)
            else:
                # NOTE: If batch_info_dict is None, we assume that
                # the tensor is not batched.
                # TODO: Set batch info in loss settings
                if loss_func.need_answer():
                    raise ValueError(f"Batch info for {key} is not found.")
                _preds = [prediction[key]]

            if loss_func.need_answer():
                _answers = unbatch(answer[key], batch_info)
            else:
                _answers = [None] * len(_preds)

            if _masks is not None:
                _preds = [
                    _select_nodes(p, m)
                    for p, m in zip(_preds, _masks, strict=True)
                ]
                _answers = [
                    a if a is None else _select_nodes(a, m)
                    for a, m in zip(_answers, _masks, strict=True)
                ]

            loss_items[loss_setting.name] = torch.mean(
                torch.stack(
                    [
                        loss_func(p, a)
                        for p, a in zip(_preds, _answers, strict=True)
                    ]
                )
            )

        return phlower_tensor_collection(loss_items)


def _unbatch_mask(
    mask_name: str, field_data: ISimulationField | None
) -> list[PhlowerTensor]:
    if field_data is None or mask_name not in field_data:
        raise ValueError(f"Mask field {mask_name} is not found.")

    mask = field_data[mask_name]
    values = mask.to_tensor()
    if ((values != 0) & (values != 1)).any():
        raise ValueError(
            f"Mask field {mask_name} must be 0 or 1. It must not be scaled."
        )
    return unbatch(mask, field_data.get_batch_info(mask_name))


def _select_nodes(tensor: PhlowerTensor, mask: PhlowerTensor) -> PhlowerTensor:
    if tensor.is_voxel:
        raise ValueError("Mask of the loss is not supported for voxel data.")
    if tuple(mask.shape) != (tensor.n_vertices(), 1):
        raise ValueError(
            f"Mask of shape {tuple(mask.shape)} does not match "
            f"the tensor of shape {tuple(tensor.shape)}."
        )

    indices = torch.nonzero(mask.to_tensor().flatten()).squeeze(-1)
    if len(indices) == 0:
        raise ValueError("No node is selected by the mask in a sample.")
    # NOTE: Indexing keeps the physical dimension of the tensor,
    # which torch.index_select does not support.
    return tensor[(slice(None),) * tensor.shape_pattern.nodes_dim + (indices,)]
