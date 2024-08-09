from __future__ import annotations

from collections.abc import Iterator

import torch

from phlower.settings import PhlowerTrainerSetting
from phlower.utils import OptimizerSelector, SchedulerSelector


class PhlowerOptimizerWrapper:
    @classmethod
    def from_setting(
        cls, setting: PhlowerTrainerSetting, model: torch.nn.Module
    ) -> PhlowerOptimizerWrapper:
        return PhlowerOptimizerWrapper(
            parameters=model.parameters(),
            optimizer=setting.optimizer_setting.optimizer,
            optimizer_kwards=setting.optimizer_setting.parameters,
            schedulers={
                v.scheduler: v.parameters for v in setting.scheduler_setting
            },
        )

    def __init__(
        self,
        parameters: Iterator[torch.nn.Parameter],
        optimizer: str,
        optimizer_kwards: dict,
        schedulers: dict[str, dict],
    ):
        self._optimizer = OptimizerSelector.select(optimizer)(
            parameters, **optimizer_kwards
        )
        self._schedulers = [
            SchedulerSelector.select(k)(self._optimizer, **v)
            for k, v in schedulers.items()
        ]

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step_optimizer(self):
        self._optimizer.step()

    def step_scheduler(self):
        for scheduler in self._schedulers:
            scheduler.step()

    def state_dict(self):
        return self._optimizer.state_dict()
