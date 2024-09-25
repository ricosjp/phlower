import functools

import numpy as np
import torch


class PhlowerTensorShapePattern:
    def __init__(self, shape: torch.Size, is_time_series: bool, is_voxel: bool):
        self._shape = shape
        self._is_time_series = is_time_series
        self._is_voxel = is_voxel

    def pattern_to_ndim(self, drop_last: bool = False) -> dict[str, int]:
        chars = self.pattern.split(" ")
        if drop_last:
            chars.pop()

        return {c: self._shape[i] for i, c in enumerate(chars)}

    def n_vertices(self) -> int:
        start = 1 if self._is_time_series else 0

        if self._is_voxel:
            return np.prod(self._shape[start : start + 3])
        return self._shape[start]

    @property
    def is_time_series(self) -> bool:
        return self._is_time_series

    @property
    def is_voxel(self) -> bool:
        return self._is_voxel

    @property
    def shape(self) -> torch.Size:
        return self._shape

    @property
    def rank_size(self) -> int:
        return len(self._shape[self.feature_start_dim : -1])

    @functools.cached_property
    def pattern(self) -> str:
        patterns = [
            self.time_series_pattern,
            self.space_pattern,
            self.feature_pattern,
        ]

        return " ".join([p for p in patterns if len(p) != 0])

    @property
    def time_series_pattern(self) -> str:
        return "t" if self._is_time_series else ""

    @property
    def space_pattern(self) -> str:
        return "x y z" if self._is_voxel else "n"

    @property
    def feature_start_dim(self) -> int:
        offset_time = 1 if self._is_time_series else 0
        offset_space = 3 if self._is_voxel else 1

        return offset_time + offset_space

    @functools.cached_property
    def feature_pattern(self) -> str:
        start = self.feature_start_dim
        return " ".join([f"a{i}" for i in range(len(self._shape[start:]))])
