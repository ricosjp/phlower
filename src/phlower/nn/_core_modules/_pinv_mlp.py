from __future__ import annotations

import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import PInvMLPSetting
from phlower.utils.exceptions import NotFoundReferenceModuleError


class PInvMLP(IPhlowerCoreModule, torch.nn.Module):
    """
    Pseudo inverse of the reference MLP layer.

    Parameters
    ----------
    reference_name: str
        Name of the reference MLP layer.

    Examples
    --------
    >>> pinv_mlp = PInvMLP(reference_name="MLP0")
    >>> pinv_mlp(data)
    """

    @classmethod
    def from_setting(cls, setting: PInvMLPSetting) -> Self:
        return cls(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        return "PInvMLP"

    @classmethod
    def need_reference(cls) -> bool:
        return True

    def __init__(self, reference_name: str, **kwards) -> None:
        super().__init__()

        self._reference_name = reference_name
        self._reference: IPhlowerCoreModule | None = None

    def get_reference_name(self) -> str:
        return self._reference_name

    def resolve(
        self, *, parent: IReadonlyReferenceGroup | None = None, **kwards
    ) -> None:
        assert parent is not None

        try:
            self._reference = parent.search_module(self._reference_name)
        except KeyError as ex:
            raise NotFoundReferenceModuleError(
                f"Reference module {self._reference_name} is not found "
                "in the same group."
            ) from ex

        self._initialize()

    def _initialize(self):
        """Initialize parameters and activations after resolve() is called."""
        self._activations = [
            _utils.ActivationSelector.select_inverse(name)
            for name in self._reference._activations[::-1]
        ]
        self._chains = self._init_pinv_chains()

    def _init_pinv_chains(self) -> list[PInvLinear]:
        name = self._reference.__class__.__name__
        if name in ["MLP", "Proportional"]:
            return self._init_pinv_mlp_chains(self._reference._chains)

        raise ValueError(f"Unsupported reference class: {name}")

    def _init_pinv_mlp_chains(
        self, chains: _utils.ExtendedLinearList
    ) -> list[PInvLinear]:
        return [PInvLinear(c) for c in chains._linears[::-1]]

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        field_data: ISimulationField | None = None,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overload torch.nn.Module

        Args:
            data: IPhlowerTensorCollections
                data which receives from predecessors
            field_data: ISimulationField | None
                Constant information through training or prediction

        Returns:
            PhlowerTensor:
                Tensor object
        """
        if self._reference is None:
            raise ValueError(
                "reference module in PInvMLP module is not set. "
                "Please check that `resolve` function is called."
            )

        h = data.unique_item()
        for activation, chain in zip(
            self._activations, self._chains, strict=True
        ):
            # Activation comes first because it is pseudo inverse
            h = chain(activation(h))

        return h


class PInvLinear(torch.nn.Module):
    def __init__(self, ref_linear: torch.nn.Linear):
        super().__init__()
        self.ref_linear = ref_linear
        self.has_bias = self.ref_linear.bias is not None
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_bias:
            return torch.nn.functional.linear(x + self.bias, self.weight)
        return torch.nn.functional.linear(x, self.weight)

    @property
    def weight(self) -> torch.Tensor:
        """Return pseudo inversed weight."""
        w = self.ref_linear.weight
        return torch.pinverse(w)

    @property
    def bias(self) -> torch.Tensor | float:
        """Return inverse bias."""
        if self.ref_linear.bias is None:
            return 0
        else:
            return -self.ref_linear.bias
