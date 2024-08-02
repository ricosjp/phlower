import abc


class IPhlowerLayerParameters(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def gather_input_dims(cls, *input_dims: int) -> int: ...

    @abc.abstractmethod
    def get_n_nodes(self) -> list[int] | None: ...

    @abc.abstractmethod
    def overwrite_nodes(self, nodes: list[int]) -> None: ...
