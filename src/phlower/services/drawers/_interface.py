import abc


class IPhlowerDrawableModule(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    def is_group(self) -> bool:
        ...

    @abc.abstractmethod
    def resolve(self) -> None: ...

    @abc.abstractmethod
    def forward(
        self,
        data: IPhlowerTensorCollections,
        *,
        supports: dict[str, PhlowerTensor]
    ) -> IPhlowerTensorCollections: ...

    @abc.abstractmethod
    def get_destinations(self) -> list[str]: ...

    @abc.abstractmethod
    def get_n_nodes(self) -> list[int]: ...

    @abc.abstractmethod
    def get_display_info(self) -> str: ...
