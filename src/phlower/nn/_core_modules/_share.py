from __future__ import annotations

from phlower._base.tensors import PhlowerTensor
from phlower._fields import ISimulationField
from phlower.collections.tensors import IPhlowerTensorCollections
from phlower.nn._interface_module import (
    IPhlowerCoreModule,
    IReadonlyReferenceGroup,
)
from phlower.settings._module_settings import ShareSetting
from phlower.utils.exceptions import NotFoundReferenceModuleError


class Share(IPhlowerCoreModule):
    """Share module have same operations as the reference module.

    Parameters
    ----------
    reference_name: str
        Name of the reference module.

    Notes
    -----
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
                "reference module in Share module is not set. "
                "Please check that `resolve` function is called."
            )

        return self._reference.forward(data, field_data=field_data, **kwards)
