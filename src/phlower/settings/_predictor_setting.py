from __future__ import annotations

import pydantic
from typing_extensions import Self

from phlower.utils.enums import ModelSelectionType


class PhlowerPredictorSetting(pydantic.BaseModel):
    selection_mode: str
    """
    Define method to select checkpoint file.
    Choose from "best", "latest", "train_best", "specified"
    """

    device: str = "cpu"
    """
    device name. Defaults to cpu
    """

    log_file_name: str = "log"
    """
    name of log file. Defaults to "log"
    """

    saved_setting_filename: str = "model"
    """
    file name of pretrained model setting. Defaults to "model"
    """

    batch_size: int = 1
    """
    batch size. Defaults to 1
    """

    num_workers: int = 0
    """
    the number of cores. Defaults to 0
    """

    non_blocking: bool = False
    """
    If True, the data transfer is non-blocking. Defaults to False.
    """

    pin_memory: bool = False
    """
    If True, the data loader uses pin_memory. Defaults to False.
    """

    random_seed: int = 0
    """
    random seed. Defaults to 0
    """

    target_epoch: int | None = None
    """
    target_epoch specifies the number of snapshot. Defaults to None.
    """

    output_to_scaler_name: dict[str, str] = pydantic.Field(default_factory=dict)
    """
    output_to_scaler_name is a dictionary to define the scaler
     for each output variable. The key is the name of the output variable
    and the value is the name of variable which has the scaler to use.
    Defaults to empty dictionary, relationship between output variable
     and scaler is assumed to be the same as that of label variables.
    """

    use_inference_mode: bool = True
    """
    If True, use torch.inference_mode() for prediction. Defaults to True.
    """

    # special keyward to forbid extra fields in pydantic
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    @pydantic.field_validator("selection_mode")
    @classmethod
    def check_valid_selection_mode(cls, name: str) -> str:
        names = [v.value for v in ModelSelectionType]
        if name not in names:
            raise ValueError(f"{name} selection mode does not exist.")
        return name

    @pydantic.model_validator(mode="after")
    def check_valid_target_epoch(self) -> Self:
        if self.selection_mode != "specified":
            if self.target_epoch is not None:
                raise ValueError(
                    "target_epoch should be None "
                    "if selection_mode is not `specified` "
                    f"but {self.target_epoch}"
                )
            return self

        if (self.target_epoch is None) or (self.target_epoch < 0):
            raise ValueError(
                "target_epoch should be non-negative value "
                f"when selection_mode = {self.selection_mode}, "
                f"but {self.target_epoch}"
            )

        return self
