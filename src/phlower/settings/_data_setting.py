import pathlib

import pydantic


class PhlowerDataSetting(pydantic.BaseModel):
    train: list[pathlib.Path] = pydantic.Field(default_factory=list)
    validation: list[pathlib.Path] = pydantic.Field(default_factory=list)
    test: list[pathlib.Path] = pydantic.Field(default_factory=list)

    model_config = pydantic.ConfigDict(frozen=True)

    def is_empty(self) -> bool:
        return not any([self.train, self.validation, self.test])
