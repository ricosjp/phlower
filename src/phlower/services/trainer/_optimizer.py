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
            optimizer_kwargs=setting.optimizer_setting.parameters,
            schedulers={
                v.scheduler: v.parameters for v in setting.scheduler_settings
            },
        )

    def __init__(
        self,
        parameters: Iterator[torch.nn.Parameter],
        optimizer: str,
        optimizer_kwargs: dict | None = None,
        schedulers: dict[str, dict] | None = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        if schedulers is None:
            schedulers = {}

        self._optimizer = OptimizerSelector.select(optimizer)(
            parameters, **optimizer_kwargs
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

    def state_dict(self) -> dict:
        return {
            "optimizer": self._optimizer.state_dict(),
            "schedulers": [
                scheduler.state_dict() for scheduler in self._schedulers
            ],
        }

    def load_state_dict(self, content: dict) -> None:
        self._optimizer.load_state_dict(content["optimizer"])

        for i, state in enumerate(content["schedulers"]):
            self._schedulers[i].load_state_dict(state)
