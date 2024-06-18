from typing import NamedTuple


class SparseBatchInfo(NamedTuple):
    sizes: list[int]
    shapes: list[tuple[int]]

    def __len__(self):
        assert len(self.sizes) == len(self.shapes)
        return len(self.sizes)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}, shapes: {self.shapes}, "
            f"sizes: {self.sizes}"
        )
