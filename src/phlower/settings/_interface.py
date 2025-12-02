from __future__ import annotations

import abc


class IReadOnlyReferenceGroupSetting(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def search_module_setting(
        self, name: str
    ) -> IPhlowerLayerParameters | IPhlowerPresetGroupParameters: ...

    @abc.abstractmethod
    def search_group_setting(
        self, name: str
    ) -> IReadOnlyReferenceGroupSetting: ...


class IModuleSetting(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_name(self) -> str: ...

    @abc.abstractmethod
    def get_input_keys(self) -> list[str]: ...

    @abc.abstractmethod
    def get_destinations(self) -> list[str]: ...

    @abc.abstractmethod
    def get_output_info(self) -> dict[str, int]: ...

    @abc.abstractmethod
    def resolve(
        self,
        *resolved_outputs: dict[str, int],
        parent: IReadOnlyReferenceGroupSetting | None = None,
    ) -> None: ...


class IPhlowerLayerParameters(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def get_nn_type(cls) -> str:
        """Return the type of neural network layer

        Returns:
            str: type of neural network layer
        """
        ...

    @abc.abstractmethod
    def gather_input_dims(self, *input_dims: int) -> int:
        """Gather feature dimensions of input tensors

        Args:
            *input_dims (list[int]): feature dimensions of input tensors

        Returns:
            int: resolved dimension of input tensors
        """
        ...

    @abc.abstractmethod
    def get_default_nodes(self, *input_dims: int) -> list[str]:
        """Return desired nodes when nodes is None
        Returns:
            list[str]: default nodes
        """
        ...

    @abc.abstractmethod
    def get_n_nodes(self) -> list[int] | None:
        """Return feature dimensions inside PhlowerLayer.

        Returns:
            list[int] | None: feature dimensions
        """
        ...

    @abc.abstractmethod
    def overwrite_nodes(self, nodes: list[int]) -> None:
        """overwrite feature dimensions by using resolved information

        Args:
            nodes (list[int]):
                feature dimensions which is resolved
                 by using status of precedent and succedent phlower modules.
        """
        ...

    @property
    @abc.abstractmethod
    def need_reference(self) -> bool:
        """Whether the reference to other phlower modules is needed or not.

        Returns:
            bool: True if reference to other phlower modules is necessary.
        """
        ...

    @abc.abstractmethod
    def get_reference(self, parent: IReadOnlyReferenceGroupSetting):
        """Get reference information from parent group setting

        Args:
            parent (IReadOnlyReferenceGroupSetting):
                Reference to group setting of its parent
        """
        ...

    @abc.abstractmethod
    def confirm(self, self_module: IModuleSetting) -> None:
        """Chenck and confirm parameters. This functions is
         called after all values of parameters are established.
         Write scripts to check its state at last.

        Args:
            input_keys (list[str]): keys to input this parameters
            **kwards: information of parent module
        """
        ...


class IPhlowerPresetGroupParameters(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def get_preset_type(cls) -> str:
        """Return the type of preset group

        Returns:
            str: type of preset group
        """
        ...

    @abc.abstractmethod
    def confirm(self, self_module: IModuleSetting) -> None:
        """Chenck and confirm parameters. This functions is
         called after all values of parameters are established.
         Write scripts to check its state at last.

        Args:
            input_keys (list[str]): keys to input this parameters
            **kwards: information of parent module
        """
        ...
