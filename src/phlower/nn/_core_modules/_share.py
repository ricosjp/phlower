from __future__ import annotations

import torch

from phlower._base.tensors import PhlowerTensor
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import ShareSetting
from phlower.utils.exceptions import NotFoundReferenceModuleError


class Share(IPhlowerCoreModule, torch.nn.Module):
    """
    Share module is a reference to another module.
    Share module itself does not have any trainable parameters.
    """

    @classmethod
    def from_setting(cls, setting: ShareSetting) -> Share:
        return Share(**setting.__dict__)

    @classmethod
    def get_nn_name(cls) -> str:
        return "Share"

    @classmethod
    def need_resolve(cls) -> bool:
        return True

    def __init__(self, reference_name: str, **kwards) -> None:
        super().__init__()

        self._reference_name = reference_name
        self._reference: IPhlowerCoreModule | None = None

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
                "reference module in Share module is not set. "
                "Please check that `resolve` function is called."
            )

        return self._reference.forward(data, supports=supports, **kwards)
