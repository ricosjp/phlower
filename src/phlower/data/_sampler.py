from __future__ import annotations

from typing import NamedTuple

import numpy as np
from phlower_tensor import IPhlowerArray, phlower_array

from phlower.settings import (
    ArrayDataIOSetting,
)

# region utility for random sampling


class _RandomSampledInfo(NamedTuple):
    n_points: int
    sampled_indices: np.ndarray


class RandomPointSampler:
    def __init__(self):
        self._history: dict[str, _RandomSampledInfo] = {}

    def reset(self) -> None:
        self._history = {}

    def sample(
        self, io_setting: ArrayDataIOSetting, array: IPhlowerArray
    ) -> IPhlowerArray:

        assert io_setting.random_sampling.is_active

        if io_setting.is_voxel:
            raise ValueError(
                "Random sampling is not supported for voxel data. "
                "Please set `random_sampling.is_active` to False "
                f"for {io_setting.name}."
            )

        # NOTE: SO far, we assume that
        # array's shape is (<n_times>, n_nodes, n_features)
        point_index = 1 if array.is_time_series else 0

        selected_indices = self._create_selected_indices(
            io_setting=io_setting, array=array, point_index=point_index
        )

        ndarray = array.to_numpy()
        if point_index == 0:
            return phlower_array(
                ndarray[selected_indices, ...],
                is_time_series=array.is_time_series,
                is_voxel=array.is_voxel,
                dimensions=array.dimension,
                dtype=ndarray.dtype,
            )
        if point_index == 1:
            return phlower_array(
                ndarray[:, selected_indices, ...],
                is_time_series=array.is_time_series,
                is_voxel=array.is_voxel,
                dimensions=array.dimension,
                dtype=ndarray.dtype,
            )

        # NOTE: It should not reach here,
        # but just in case, we raise an error.
        raise ValueError(
            "Unexpected error in random sampling. "
            f"n_node_index: {point_index}, array shape: {array.shape}"
        )

    def _create_selected_indices(
        self,
        io_setting: ArrayDataIOSetting,
        array: IPhlowerArray,
        point_index: int,
    ) -> np.ndarray:

        key = io_setting.random_sampling.same_as or io_setting.name

        if key in self._history:
            hist = self._history[key]
            n_nodes = array.shape[point_index]
            if hist.n_points != n_nodes:
                raise ValueError(
                    f"Number of points does not match for {io_setting.name}."
                    f" Expected n points: {hist.n_points}, Actual: {n_nodes}, "
                    f"{io_setting.random_sampling.same_as=}"
                )
            return hist.sampled_indices

        n_nodes = array.shape[point_index]
        selected_indices = np.random.choice(
            n_nodes,
            size=io_setting.random_sampling.n_sampled_points,
            replace=False,
        )
        # store in history
        self._history[key] = _RandomSampledInfo(
            n_points=n_nodes,
            sampled_indices=selected_indices,
        )
        return selected_indices


# endregion
