from typing import Iterable, Any

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


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class ModelSetting:
    ...


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class LayerSetting:
    type: str
    input_keys: list[str]
    output_key: str
    destinations: list[str]
    parameters: dict[str, Any]


@dc.dataclass(frozen=True, config=pydantic.ConfigDict(extra="forbid"))
class LossSettings:
    name2loss: dict[str, str]
    name2weight: dict[str, str] | None = None

    def loss_names(self) -> Iterable[str]:
        return self.name2loss.values()
