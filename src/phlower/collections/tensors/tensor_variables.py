from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
import torch

BuiltInVars = TypeVar("BuiltInVars", torch.TensorType, Sequence, Mapping)


class IPhlowerVariables:
    def __init__(self, value: BuiltInVars) -> None:
        raise NotImplementedError()

    def get_values(self) -> BuiltInVars:
        raise NotImplementedError()

    def slice(self, loss_slice: slice) -> IPhlowerVariables:
        raise NotImplementedError()

    def send(self, device: str) -> IPhlowerVariables:
        raise NotImplementedError()

    def min_len(self) -> int:
        raise NotImplementedError()

    def to_numpy(self) -> Any:
        raise NotImplementedError()


def phlower_tensor_variables(values: BuiltInVars) -> IPhlowerVariables:
    if isinstance(values, torch.Tensor):
        return PhlowerTensorVariables(values)
    if isinstance(values, dict):
        return PhlowerDictTensors(values)
    if isinstance(values, list):
        return PhlowerListTensors(values)

    if isinstance(values, PhlowerListTensors):
        return PhlowerListTensors(values.get_values())
    if isinstance(values, PhlowerDictTensors):
        return PhlowerDictTensors(values.get_values())
    if isinstance(values, PhlowerTensorVariables):
        return PhlowerTensorVariables(values.get_values())

    raise NotImplementedError(
        f"Converter for data type: {type(values)} is not implemented"
    )


# HACK This is integrated to PhlowerTensor
class PhlowerTensorVariables(IPhlowerVariables):
    def __init__(self, value: torch.Tensor):
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                "cannot initialize objects except torch.Tensor."
                f" Input: {type(value)}"
            )
        self._x = value

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {str(self._x)}"

    def get_values(self) -> torch.Tensor:
        return self._x

    def slice(self, loss_slice: slice) -> PhlowerTensorVariables:
        tmp = self._x[loss_slice]
        return PhlowerTensorVariables(tmp)

    def send(self, device: str) -> PhlowerTensorVariables:
        tmp = self._x.to(device)
        return PhlowerTensorVariables(tmp)

    def min_len(self) -> int:
        return len(self._x)

    def to_numpy(self) -> Any:
        return self._x.cpu().detach().numpy()


class PhlowerDictTensors(IPhlowerVariables):
    def __init__(self, value: dict):
        self._x = {k: phlower_tensor_variables(v) for k, v in value.items()}

    def __str__(self) -> str:
        txt = ""
        for k, v in self._x.items():
            txt += f"{k}: {v}, "
        return f"{self.__class__.__name__}" + "{" + txt + "}"

    def get_values(self) -> dict[str, torch.Tensor]:
        tmp = {k: v.get_values() for k, v in self._x.items()}
        return tmp

    def slice(self, loss_slice: slice) -> PhlowerDictTensors:
        tmp = {k: v.slice(loss_slice) for k, v in self._x.items()}
        return PhlowerDictTensors(tmp)

    def send(self, device: str) -> PhlowerTensorVariables:
        tmp = {k: v.send(device) for k, v in self._x.items()}
        return PhlowerDictTensors(tmp)

    def min_len(self) -> int:
        return np.min([x.min_len() for x in self._x.values()])

    def to_numpy(self) -> Any:
        return {k: v.to_numpy() for k, v in self._x.items()}


class PhlowerListTensors(IPhlowerVariables):
    def __init__(self, value: list[torch.Tensor]):
        self._x = [phlower_tensor_variables(v) for v in value]

    def __str__(self) -> str:
        values = [str(v) for v in self._x]
        txt = ", ".join(values)
        return f"{self.__class__.__name__}" + "[" + txt + "]"

    def get_values(self) -> list[torch.Tensor]:
        tmp = [v.get_values() for v in self._x]
        return tmp

    def slice(self, loss_slice: slice) -> PhlowerListTensors:
        tmp = [v.slice(loss_slice) for v in self._x]
        return PhlowerListTensors(tmp)

    def send(self, device: str) -> PhlowerListTensors:
        tmp = [v.send(device) for v in self._x]
        return PhlowerListTensors(tmp)

    def min_len(self) -> int:
        return np.min([x.min_len() for x in self._x])

    def to_numpy(self) -> Any:
        return [v.to_numpy() for v in self._x]
