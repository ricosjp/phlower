import pydantic

from ._interface import IContinueParameter, IReadOnlyContinueSetting


class LRDecayContinueSetting(pydantic.BaseModel, IContinueParameter):
    lr_factor: float = pydantic.Field(lt=1, gt=0)
    """
    Learning rate decay factor. Learning rate for next training is calculated
     as `original_lr * (lr_factor ** continue_count)`.
    """

    def validate(self, parent: IReadOnlyContinueSetting) -> None:
        pass

    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )
