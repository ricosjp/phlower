from __future__ import annotations

import abc


class IReadOnlyReferenceGroupSetting(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def search_module_setting(self, name: str) -> IPhlowerLayerParameters: ...


class IModuleSetting(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_name(self) -> str: ...

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
    @abc.abstractmethod
    def gather_input_dims(self, *input_dims: int) -> int: ...

    @abc.abstractmethod
    def get_n_nodes(self) -> list[int] | None: ...

    @abc.abstractmethod
    def overwrite_nodes(self, nodes: list[int]) -> None: ...

    @property
    @abc.abstractmethod
    def need_reference(self) -> bool: ...

    @abc.abstractmethod
    def get_reference(self, parent: IReadOnlyReferenceGroupSetting): ...
