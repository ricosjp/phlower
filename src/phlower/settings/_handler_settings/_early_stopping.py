import pydantic
from typing_extensions import Self


class EarlyStoppingSetting:
    handler: str
    """
    handler Class name defined in phlower.services.trainer.handlers.
    """

    parameters: dict[str, int | float | bool | str] = pydantic.Field(
        default_factory=dict
    )
    """
    Parameters to pass when handler class is initialized.
    Allowed parameters depend on the handler you choose.
    """

    def get_patience(self) -> float:
        return self.parameters["patience"]

    @pydantic.model_validator(mode="after")
    def check_args(self) -> Self:
        if "patience" not in self.parameters:
            raise ValueError(
                "patience keyword is necessary to define EarlyStopping."
            )

        return self

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
