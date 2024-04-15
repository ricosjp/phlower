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
