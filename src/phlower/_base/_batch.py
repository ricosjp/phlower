from functools import cached_property
from typing import NamedTuple


class GraphBatchInfo(NamedTuple):
    sizes: list[int]
    shapes: list[tuple[int]]
    n_nodes: tuple[int]
    dense_concat_dim: int | None = None

    @cached_property
    def total_n_nodes(self) -> int:
        return sum(self.n_nodes)

    @property
    def is_concatenated(self) -> bool:
        return len(self.sizes) > 1

    def __len__(self):
        assert len(self.sizes) == len(self.shapes)
        return len(self.sizes)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}, shapes: {self.shapes}, "
            f"sizes: {self.sizes}"
        )
