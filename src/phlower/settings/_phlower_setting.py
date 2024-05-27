from __future__ import annotations

from typing import Iterable

import pydantic
from pydantic import dataclasses as dc


@dc.dataclass(frozen=True)
class PhlowerSetting:
    data_setting: dict


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class PhlowerTrainerSetting:
    random_seed: int = 0
    batch_size: int = 1
    num_workers: int = 1
    device: str = "cpu"
    non_blocking: bool = False

    variable_dimensions: dict[str, dict[str, float]] = pydantic.Field(
        default_factory=lambda: {}
    )


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class LossSettings:
    name2loss: dict[str, str]
    name2weight: dict[str, str] | None = None

    def loss_names(self) -> Iterable[str]:
        return self.name2loss.values()
