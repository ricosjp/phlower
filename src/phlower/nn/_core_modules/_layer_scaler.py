from __future__ import annotations

from typing import Self

import torch
from phlower_tensor import ISimulationField, PhlowerTensor
from phlower_tensor.collections import IPhlowerTensorCollections

from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import (
    LayerScalerSetting,
    LayerScalingMethod,
)


class LayerScaler(IPhlowerCoreModule, torch.nn.Module):
    """LayerScaler is a neural network module that performs a scaling operation
    on the input tensor.

    Parameters
    ----------
    scaling_method: str
        The method of scaling to apply. Supported methods are:
        - "signed_log": Applies the signed logarithm function to the input.
        - "asinh": Applies the inverse hyperbolic sine function to the input.

    Examples
    --------
    >>> layer_scaler = LayerScaler(
    ...     scaling_method="signed_log"
    ... )
    >>> layer_scaler(data)
    """

    @classmethod
    def from_setting(cls, setting: LayerScalerSetting) -> Self:
        """Create LayerScaler from setting object

        Args:
            setting: LayerScalerSetting
                setting object

        Returns:
            Self: LayerScaler
        """
        return LayerScaler(**setting.model_dump(exclude={"nodes"}))

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of LayerScaler

        Returns:
            str: name
        """
        return "LayerScaler"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(self, scaling_method: str):
        super().__init__()
        self._scaling_method = scaling_method

        # NOTE: After drop Python 3.11, we can use `xxx in StrEnum`
        assert self._scaling_method in LayerScalingMethod.__members__, (
            f"Unsupported scaling method: {self._scaling_method}"
        )

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None: ...

    def get_reference_name(self) -> str | None:
        return None

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overloads torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor: Tensor object
        """

        val = data.unique_item()

        match self._scaling_method:
            case LayerScalingMethod.signed_log1p:
                return _signed_log_scaling(val)
            case LayerScalingMethod.asinh:
                return _asinh_scaling(val)
            case _:
                raise ValueError(
                    f"Unsupported scaling method: {self._scaling_method}"
                )


def _signed_log_scaling(x: PhlowerTensor) -> PhlowerTensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _asinh_scaling(x: PhlowerTensor) -> PhlowerTensor:
    return torch.asinh(x)
