from __future__ import annotations

from typing import Any

import numpy as np
import torch

from phlower.utils.exceptions import PhlowerIncompatibleTensorError


class PhlowerShapePattern:
    @classmethod
    def from_pattern(
        cls, shape: torch.Size, pattern: str | PhlowerShapePattern
    ) -> PhlowerShapePattern:
        if isinstance(pattern, PhlowerShapePattern):
            str_pattern = pattern.get_pattern()
        else:
            str_pattern = pattern
        _splited = _split_pattern(str_pattern)
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

    def resolve_index_access(
        self, index: Any, to_shape: torch.Size
    ) -> PhlowerShapePattern:
        if (not self.is_time_series) and (not self.is_voxel):
            return PhlowerShapePattern(to_shape, False, False)

        if isinstance(index, torch.Tensor) and index.dtype == torch.bool:
            return PhlowerShapePattern(to_shape, False, False)

        wrapped = _IndexKeyWrapper(keys=index, shapes=self._shape)
        time_series = (
            wrapped.check_dimension_kept(0) if self.is_time_series else False
        )
        voxel = (
            wrapped.check_dimension_kept(self.start_space_index)
            and wrapped.check_dimension_kept(self.start_space_index + 1)
            and wrapped.check_dimension_kept(self.start_space_index + 2)
            if self.is_voxel
            else False
        )
        return PhlowerShapePattern(
            to_shape, is_time_series=time_series, is_voxel=voxel
        )

    def rearrange(
        self, pattern: str, to_shape: torch.Size
    ) -> PhlowerShapePattern:
        """Transform the shape pattern to a new pattern.

        Args:
            pattern (str): The new pattern string.

        Returns:
            PhlowerShapePattern: A new instance with the transformed pattern.
        """

        from_patterns = _split_pattern(pattern.split("->")[0].strip())
        to_pattern = pattern.split("->")[1].strip()

        # Fill ellipse if it exists
        if "..." in to_pattern:
            assert "..." in from_patterns

            start_idx = -1
            end_idx = -1
            for i, c in enumerate(from_patterns):
                if c != "...":
                    continue

                start_idx = i
                _n_left = len(from_patterns) - i - 1
                end_idx = len(self._shape) - _n_left
                break

            ellipse_pattern = self.get_pattern().split(" ")[start_idx:end_idx]
            to_pattern = to_pattern.replace("...", " ".join(ellipse_pattern))

        return PhlowerShapePattern.from_pattern(to_shape, to_pattern)

    def get_pattern_to_size(self, drop_last: bool = False) -> dict[str, int]:
        """Return mapping of which key is pattern symbol and
          value is its dimension value.

        Args:
            drop_last (bool, optional): If True, drop information
              on last pattern.
              Defaults to False.

        Returns:
            dict[str, int]: _description_
        """
        chars = self.get_pattern().split(" ")
        if drop_last:
            chars.pop()

        return {c: self._shape[i] for i, c in enumerate(chars)}

    def get_n_vertices(self) -> int:
        """Get the number of nodes

        Returns:
            int: the number of nodes
        """
        start = 1 if self._is_time_series else 0

        if self._is_voxel:
            return np.prod(self._shape[start : start + 3])
        return self._shape[start]

    def get_space_pattern(self, omit_space: bool = False) -> str:
        """Get space pattern.

        Args:
            omit_space (bool, optional): If True, omit space between symbols.
              Defaults to False.

        Returns:
            str: pattern which corresponds to space information.
        """

        if not self._is_voxel:
            return self.n_nodes_pattern_symbol

        if omit_space:
            return "xyz"

        return "x y z"

    def get_pattern(self) -> str:
        """Return pattern string of tensor shape

        Returns:
            str: pattern of tensor shape
        """
        patterns = [
            self.time_series_pattern,
            self.get_space_pattern(),
            self.get_feature_pattern(),
        ]

        new_pattern = " ".join([p for p in patterns if len(p) != 0])
        return new_pattern

    def get_feature_pattern(self, drop_last: bool = False) -> str:
        """Return pattern related only with features.
        Ex. time node d0 d1 ... f
                      ^^^^^^^^^^^

        Args:
            drop_last (bool, optional): If True, drop last index of shape.
              Defaults to False.

        Returns:
            str: Pattern related only with features.
        """
        start = self.feature_start_dim
        if start >= len(self._shape):
            return ""

        # dimension ranks
        _items = [f"d{i}" for i in range(len(self._shape[start:-1]))]
        if not drop_last:
            _items.append(self.feature_pattern_symbol)
        return " ".join(_items)

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
    def feature_pattern_symbol(self) -> str:
        return "f"

    @property
    def n_nodes_pattern_symbol(self) -> str:
        return "n"

    @property
    def nodes_dim(self) -> int:
        """
        Return the location index of nodes in shape list
        """
        if self._is_voxel:
            raise ValueError("n_nodes dimension does not exist.")

        return 1 if self._is_time_series else 0

    @property
    def feature_start_dim(self) -> int:
        offset_time = 1 if self._is_time_series else 0
        offset_space = 3 if self._is_voxel else 1

        return offset_time + offset_space

    def is_global(self, n_batch: int) -> bool:
        """
        Returns True if the batched tensor is from global ones.

        Args:
            n_batch: int
                The number of batches.

        Returns:
            bool: True if the batched tensor is from global ones.
        """
        return self.get_n_vertices() == n_batch

    def squeeze(self, dim: int | None = None) -> PhlowerShapePattern:
        """Return a new PhlowerShapePattern with squeezed shape.

        Returns:
            PhlowerShapePattern: A new instance with squeezed shape.
        """

        if dim is None:
            indexes = [slice(None) if s != 1 else 0 for s in self._shape]
            to_shape = torch.Size([s for s in self._shape if s != 1])
        else:
            indexes = [
                slice(None) if (i != dim) or (s != 1) else 0
                for i, s in enumerate(self._shape)
            ]
            to_shape = torch.Size(
                [
                    s
                    for i, s in enumerate(self._shape)
                    if not (i == dim and s == 1)
                ]
            )

        return self.resolve_index_access(indexes, to_shape=to_shape)


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
    if not patterns:
        return False
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


_ACCESS_ITEM = int | slice | torch.Tensor | np.ndarray | None


class _IndexKeyWrapper:
    def __init__(self, keys: _ACCESS_ITEM, shapes: torch.Size):
        keys = keys if isinstance(keys, list | tuple) else [keys]
        self._key = self._resolve_ellipse(keys, shapes)
        self._shapes = shapes

        self._first_none_index = self._find_first_none_index()

    def _find_first_none_index(self) -> int:
        for i, k in enumerate(self._key):
            if k is None:
                return i
        return np.inf

    def check_dimension_kept(self, index: int) -> bool:
        if index >= self._first_none_index:
            return False

        key = self[index]

        match key:
            case int():
                return False
            case slice():
                count = len(range(self._shapes[index])[key])
                return count >= 1
            case torch.Tensor():
                return key.numel() > 1
            case np.ndarray():
                return key.size > 1
            case None:
                return False
            case list() | tuple():
                return len(key) > 1
            case _:
                raise ValueError(
                    f"Unsupported key type for index access. {type(key)=}"
                )

    def __getitem__(
        self, index: int
    ) -> int | slice | torch.Tensor | np.ndarray:
        if index >= len(self._key):
            return slice(None)

        return self._key[index]

    def _resolve_ellipse(
        self, keys: list[_ACCESS_ITEM], shapes: torch.Size
    ) -> list[_ACCESS_ITEM]:
        total_wo_none = sum(1 for k in keys if k is not None)
        _resolved: list[_ACCESS_ITEM] = []

        for k in keys:
            if k is Ellipsis:
                n_wo_none = sum(1 for k in _resolved if k is not None)
                _resolved += [slice(None)] * (
                    len(shapes) - n_wo_none - (total_wo_none - n_wo_none - 1)
                )
                continue

            _resolved.append(k)

        return _resolved
