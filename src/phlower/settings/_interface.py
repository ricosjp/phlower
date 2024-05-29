import abc


class IPhlowerLayerParameters(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def gather_input_dims(cls, *input_dims: int) -> int: ...

    @abc.abstractmethod
    def get_nodes(self) -> list[int] | None: ...

    @abc.abstractmethod
    def overwrite_nodes(self, nodes: list[int]) -> None: ...
