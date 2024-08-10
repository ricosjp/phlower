from __future__ import annotations

import torch
from typing_extensions import Self

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import PInvMLPSetting
from phlower.utils.exceptions import (
    NotFoundReferenceModuleError,
    PhlowerInvalidActivationError,
)


class PInvMLP(IPhlowerCoreModule, torch.nn.Module):
    """
    Pseudo inverse of the reference MLP layer.
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
        self._activation_names = [
            self._inverse_activation_name(a)
            for a in self._reference._activations[::-1]]
        self._activations = [
            _utils.ActivationSelector.select(name)
            for name in self._activation_names]
        self._chains = self._init_pinv_chains()

    def _inverse_activation_name(self, activation_name):
        if activation_name == "identity":
            return "identity"
        if activation_name == "leaky_relu0p5":
            return "inversed_leaky_relu0p5"
        if activation_name == "smooth_leaky_relu":
            return "inversed_smooth_leaky_relu"
        if activation_name == "tanh":
            return "truncated_atanh"

        raise PhlowerInvalidActivationError(
            f"Cannot pinv for {activation_name}")

    def _init_pinv_chains(self):
        name = self._reference.__class__.__name__
        if name in ["MLP", "Proportional"]:
            return self._init_pinv_mlp_chains(self._reference._chains)

        raise ValueError(f"Unsupported reference class: {name}")

    def _init_pinv_mlp_chains(
            self, chains: _utils.ExtendedLinearList, option=None):
        return [PInvLinear(c, option=option) for c in chains._linears[::-1]]

    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor] | None = None,
        **kwards,
    ) -> PhlowerTensor:
        """forward function which overload torch.nn.Module

        Args:
            data (IPhlowerTensorCollections):
                data which receives from predecessors
            supports (dict[str, PhlowerTensor]):
                sparse tensor objects

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
                self._activations, self._chains, strict=True):
            # Activation comes first because it is pseudo inverse
            h = chain(activation(h))

        return h


class PInvLinear(torch.nn.Module):

    def __init__(self, ref_linear: torch.nn.Linear, option: str | None = None):
        super().__init__()
        self.ref_linear = ref_linear
        self.option = option
        return

    def forward(self, x):
        h = torch.nn.functional.linear(x + self.bias, self.weight)
        return h

    @property
    def weight(self):
        """Return pseudo inversed weight."""
        if self.option is None:
            w = self.ref_linear.weight
        else:
            raise ValueError(f"Unexpected option: {self.option}")
        return torch.pinverse(w)

    @property
    def bias(self):
        """Return inverse bias."""
        if self.ref_linear.bias is None:
            return 0
        else:
            return - self.ref.bias
