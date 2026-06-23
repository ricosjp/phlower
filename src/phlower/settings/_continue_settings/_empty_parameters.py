import pydantic

from ._interface import IContinueParameter, IReadOnlyContinueSetting


class EmptyContinueSetting(pydantic.BaseModel, IContinueParameter):
    """
    This class is used to represent an empty parameter setting.
    """

    def validate(self, parent: IReadOnlyContinueSetting) -> None:
        pass

    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )
