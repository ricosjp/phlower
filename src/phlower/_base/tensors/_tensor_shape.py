from __future__ import annotations

import numpy as np
import torch

from phlower.utils.exceptions import PhlowerIncompatibleTensorError


class PhlowerShapePattern:
    @classmethod
    def from_pattern(
        cls, shape: torch.Size, pattern: str
    ) -> PhlowerShapePattern:
        _splited = _split_pattern(pattern)
        if not _check_shape_and_pattern(shape, _splited):
            raise PhlowerIncompatibleTensorError(
                "Invalid tensor shape and pattern. "
                f"shape: {shape}, pattern: {pattern}"
            )

        is_time_series = _check_is_time_series(_splited)
        is_voxel = _check_is_voxel(_splited, is_time_series)
        return PhlowerShapePattern(shape, is_time_series, is_voxel)

    def __init__(
        self,
        shape: torch.Size,
        is_time_series: bool,
        is_voxel: bool,
    ):
        self._shape = shape
        self._is_time_series = is_time_series
        self._is_voxel = is_voxel

    def get_pattern_to_size(self, drop_last: bool = False) -> dict[str, int]:
        chars = self.get_pattern().split(" ")
        if drop_last:
            chars.pop()

        return {c: self._shape[i] for i, c in enumerate(chars)}

    def get_n_vertices(self) -> int:
        start = 1 if self._is_time_series else 0

        if self._is_voxel:
            return np.prod(self._shape[start : start + 3])
        return self._shape[start]

    def get_space_pattern(self, omit_space: bool = False) -> str:
        if not self._is_voxel:
            return "n"

        if omit_space:
            return "xyz"

        return "x y z"

    def get_pattern(self) -> str:
        patterns = [
            self.time_series_pattern,
            self.get_space_pattern(),
            self.get_feature_pattern(),
        ]

        new_pattern = " ".join([p for p in patterns if len(p) != 0])
        return new_pattern

    def get_feature_pattern(self) -> str:
        start = self.feature_start_dim
        return " ".join([f"a{i}" for i in range(len(self._shape[start:]))])

    def __str__(self):
        return f"ShapePattern: {self.get_pattern()}"

    @property
    def start_space_index(self) -> int:
        return 1 if self._is_time_series else 0

    @property
    def space_width(self) -> int:
        return 3 if self.is_voxel else 1

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

    @property
    def time_series_pattern(self) -> str:
        return "t" if self._is_time_series else ""

    @property
    def feature_start_dim(self) -> int:
        offset_time = 1 if self._is_time_series else 0
        offset_space = 3 if self._is_voxel else 1

        return offset_time + offset_space


def _check_shape_and_pattern(shape: torch.Size, patterns: list[str]) -> bool:
    if len(shape) == len(patterns):
        return True

    if len(shape) < len(patterns):
        return False

    contain_ellipse = np.any([("..." in p) for p in patterns])
    if contain_ellipse:
        return True
    else:
        return False


def _check_is_time_series(patterns: list[str]) -> bool:
    return _match_to_one_word(patterns[0], "t")


def _check_is_voxel(patterns: list[str], is_time_series: bool) -> bool:
    offset = 1 if is_time_series else 0

    if len(patterns) < offset + 3:
        return False

    is_x = _match_to_one_word(patterns[offset], "x")
    is_y = _match_to_one_word(patterns[offset + 1], "y")
    is_z = _match_to_one_word(patterns[offset + 2], "z")

    return is_x and is_y and is_z


def _match_to_one_word(target: str, char: str) -> bool:
    if len(target) == 1:
        return target == char

    if target.startswith("(") and target.endswith(")"):
        _collect = "".join(target[1:-1].split())
        return _collect == char

    return False


def _split_pattern(pattern: str) -> list[str]:
    splited: list[str] = []
    index = 0
    N_ = len(pattern)

    while index < N_:
        if pattern[index] == "(":
            p, index = _collect_until_brace_end(pattern, index)
            splited.append(p)
            continue

        if pattern[index] == " ":
            index += 1
            continue

        if pattern[index] == ".":
            if pattern[index : index + 3] == "...":
                splited.append("...")
                index += 3
                continue
            raise ValueError(f"Invalid Ellipse found. {pattern}")

        if pattern[index].isalpha():
            splited.append(pattern[index])
            index += 1
            continue

        if pattern[index].isnumeric():
            if index == 0:
                raise ValueError(
                    "pattern starting with numerics is invalid. "
                    f"pattern: {pattern}"
                )
            splited[-1] += pattern[index]
            index += 1
            continue

        raise ValueError(f"Unknown pattern: {pattern=}")
    return splited


def _collect_until_brace_end(pattern: str, start: int) -> tuple[str, int]:
    index = start
    if pattern[index] != "(":
        raise ValueError(f"{pattern[index:]} is not start with ( .")

    count = 0
    while index < len(pattern):
        if pattern[index] == "(":
            count += 1

        if pattern[index] == ")":
            count -= 1
            if count == 0:
                return pattern[start : index + 1], index + 1

        index += 1

    raise ValueError(f"brace is not closed correctly. {pattern}")
