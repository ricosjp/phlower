from __future__ import annotations

import torch

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._core_modules import _utils
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import DirichletSetting


class Dirichlet(IPhlowerCoreModule, torch.nn.Module):
    """Dirichlet is a neural network module that overwrites values
    with that of dirichlet field.

    Parameters
    ----------
    activation: str
        Name of the activation function to apply to the output.
    dirichlet_name: str
        Name of the dirichlet field.
    nodes: list[int] | None (optional)
        List of feature dimension sizes (The last value of tensor shape).
        Defaults to None.

    Examples
    --------
    >>> dirichlet = Dirichlet(activation="relu", dirichlet_name="dirichlet")
    >>> dirichlet(data)

    """

    @classmethod
    def from_setting(cls, setting: DirichletSetting) -> Dirichlet:
        """Create Dirichlet from setting object

        Args:
            setting (DirichletSetting): setting object

        Returns:
            Self: Dirichlet
        """
        return Dirichlet(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        """Return name of Dirichlet

        Returns:
            str: name
        """
        return "Dirichlet"

    @classmethod
    def need_reference(cls) -> bool:
        return False

    def __init__(
        self, activation: str, dirichlet_name: str, nodes: list[int] = None
    ):
        super().__init__()
        self._nodes = nodes
        self._activation_name = activation
        self._activation_func = _utils.ActivationSelector.select(activation)
        self._dirichlet_name = dirichlet_name

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
            supports: dict[str, PhlowerTensor] | None
                Graph object. Defaults to None. Dirichlet will not use it.

        Returns:
            PhlowerTensor: Tensor object
        """
        dirichlet = data[self._dirichlet_name]
        value_names = list(
            filter(lambda name: name != self._dirichlet_name, data.keys())
        )
        if len(value_names) != 1:
            raise ValueError(
                "Dirichlet value cannot be detected. "
                f"Candidates: {value_names}"
            )

        value_name = value_names[0]
        dirichlet_filter = torch.isnan(dirichlet)
        ans = torch.where(dirichlet_filter, data[value_name], dirichlet)
        return self._activation_func(ans)
