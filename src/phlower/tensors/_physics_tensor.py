from __future__ import annotations

import torch

from enum import Enum

# class ISimlVariables():
#     def __init__(self, value: BuiltInVars) -> None:
#         raise NotImplementedError()

#     def get_values(self) -> BuiltInVars:
#         raise NotImplementedError()

#     def slice(self, loss_slice: slice) -> ISimlVariables:
#         raise NotImplementedError()

#     def send(self, device: str) -> ISimlVariables:
#         raise NotImplementedError()

#     def min_len(self) -> int:
#         raise NotImplementedError()

#     def to_numpy(self) -> Any:
#         raise NotImplementedError()


class PhysicsTensor:
    def __init__(self, tensor: torch.Tensor):
        self._is_time_series: bool = False
        self._tensor = tensor

    def __str__(self) -> str:
        return f"PhysicsTensor: {self._tensor}"

    def __add__(self, other) -> PhysicsTensor:
        return torch.add(self, other)

    @classmethod
    def __torch_function__(
        cls,
        func: callable,
        types: list[type],
        args: tuple,
        kwargs: dict | None = None,
    ):
        if kwargs is None:
            kwargs = {}

        args = [getattr(a, "_tensor", a) for a in args]
        ret: torch.Tensor = func(*args, **kwargs)
        return PhysicsTensor(ret)
