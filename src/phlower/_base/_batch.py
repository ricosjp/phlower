from typing import NamedTuple


class BatchInfo(NamedTuple):
    sizes: list[int]
    shapes: list[tuple[int]]

    @property
    def is_concatenated(self) -> bool:
        return len(self.sizes) > 1

    def get_total_nodes(self) -> int:
        return sum(v[0] for v in self.shapes)

    def __len__(self):
        assert len(self.sizes) == len(self.shapes)
        return len(self.sizes)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}, shapes: {self.shapes}, "
            f"sizes: {self.sizes}"
        )
