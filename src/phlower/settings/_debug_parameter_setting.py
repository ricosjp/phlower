import pydantic
from pydantic import Field


class PhlowerModuleDebugParameters(pydantic.BaseModel):
    output_tensor_shape: list[int] | None = Field(None, frozen=True)
    """
    If not None, you can check the shape of the output tensor of the module
    during the forward pass.
    Shape is specified as a list of integers, where -1 matches any size.
    """

    dump_forward_tensor: bool = Field(False, frozen=True)
    """
    If True, the output tensor of the module will be dumped
    during the forward pass. The tensor will be saved in the output directory
    specified by training settings.
    """

    dump_backward_tensor: bool = Field(False, frozen=True)
    """
    If True, the gradient tensor of the output tensor will be dumped
    during the backward pass. The tensor will be saved in the output directory
    specified by training settings.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
