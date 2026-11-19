import abc
from typing import Any, ClassVar, Literal

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
)


class IInitParameter(metaclass=abc.ABCMeta):
    """Initialization strategy for a linear layer."""

    @abc.abstractmethod
    def apply(self, layer: torch.nn.Linear) -> None:
        """Initialize ``layer`` in place.

        Parameters
        ----------
        layer : torch.nn.Linear
            Linear layer to initialize.
        """

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert this initialization to built-in values.

        Returns
        -------
        dict[str, Any]
            Parameters of this initialization, including ``kind``.
        """


class _TruncatedNormalInitParameters(BaseModel, IInitParameter):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ClassVar[str] = "truncated_normal"
    std: float = Field(0.02, gt=0, allow_inf_nan=False, frozen=True)

    def apply(self, layer: torch.nn.Linear) -> None:
        torch.nn.init.trunc_normal_(layer.weight, std=self.std)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.model_dump()}


class _XavierUniformInitParameters(BaseModel, IInitParameter):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: ClassVar[str] = "xavier_uniform"
    gain: float = Field(1.0, allow_inf_nan=False, frozen=True)

    def apply(self, layer: torch.nn.Linear) -> None:
        torch.nn.init.xavier_uniform_(layer.weight, gain=self.gain)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.model_dump()}


class _XavierNormalInitParameters(BaseModel, IInitParameter):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: ClassVar[str] = "xavier_normal"
    gain: float = Field(1.0, allow_inf_nan=False, frozen=True)

    def apply(self, layer: torch.nn.Linear) -> None:
        torch.nn.init.xavier_normal_(layer.weight, gain=self.gain)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.model_dump()}


_INIT_PARAMETER_MODELS = {
    model.kind: model
    for model in (
        _TruncatedNormalInitParameters,
        _XavierUniformInitParameters,
        _XavierNormalInitParameters,
    )
}


class LinearInitSetting(BaseModel):
    """Optional weight initialization and bias zeroing for linear layers.

    The initializer kind selects the schema for ``init_parameters``.
    Truncated normal uses ``std`` (default 0.02). Both Xavier methods use
    ``gain`` (default 1.0).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["truncated_normal", "xavier_uniform", "xavier_normal"] = (
        Field("truncated_normal", frozen=True)
    )
    init_parameters: (
        _TruncatedNormalInitParameters
        | _XavierUniformInitParameters
        | _XavierNormalInitParameters
    ) = Field(default_factory=dict, validate_default=True, frozen=True)
    zero_bias: bool = Field(True, frozen=True)

    @field_validator("init_parameters", mode="before")
    @classmethod
    def _resolve_init_parameters(
        cls, parameters: object, info: ValidationInfo
    ) -> object:
        parameter_type = _INIT_PARAMETER_MODELS.get(info.data.get("kind"))
        if parameter_type is None:
            return parameters
        return parameter_type.model_validate(parameters)

    def apply(self, layer: torch.nn.Linear) -> None:
        """Initialize linear layers in place.

        Parameters
        ----------
        layer : torch.nn.Linear
            Linear layer to initialize.
        """
        self.init_parameters.apply(layer)
        if self.zero_bias and layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

    def to_dict(self) -> dict[str, Any]:
        parameters = self.init_parameters.to_dict()
        kind = parameters.pop("kind")
        return {
            "kind": kind,
            "init_parameters": parameters,
            "zero_bias": self.zero_bias,
        }
