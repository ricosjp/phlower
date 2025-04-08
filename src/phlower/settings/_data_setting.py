from __future__ import annotations

import pathlib

import pydantic


class PhlowerDataSetting(pydantic.BaseModel):
    training: list[pathlib.Path] = pydantic.Field(default_factory=list)
    validation: list[pathlib.Path] = pydantic.Field(default_factory=list)

    model_config = pydantic.ConfigDict(frozen=True)

    def is_empty(self) -> bool:
        return len(self.training) == 0 and len(self.validation) == 0

    def upgrade(
        self,
        training: list[pathlib.Path] | None = None,
        validation: list[pathlib.Path] | None = None,
    ) -> PhlowerDataSetting:
        if training is None and validation is None:
            return self

        _overwrite = {
            "training": training if training else self.training,
            "validation": validation if validation else self.validation,
        }
        return PhlowerDataSetting(**_overwrite)
