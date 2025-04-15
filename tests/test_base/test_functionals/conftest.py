from collections.abc import Callable

import numpy as np
import pytest
import torch
from phlower._base import PhysicalDimensions, phlower_tensor
from phlower._base.array import phlower_array
from phlower._base.tensors._interface import IPhlowerTensor
from scipy import sparse as sp


@pytest.fixture
def create_sparse_tensors() -> (
    Callable[
        [list[tuple[int]], list[dict[str, float]] | None], list[IPhlowerTensor]
    ]
):
    def _create(
        shapes: list[tuple], dimensions: list[dict] | None = None
    ) -> list[IPhlowerTensor]:
        rng = np.random.default_rng()
        if dimensions is None:
            return [
                phlower_tensor(
                    phlower_array(
                        sp.random(
                            shape[0], shape[1], density=0.1, random_state=rng
                        ),
                    ).to_tensor()
                )
                for shape in shapes
            ]

        return [
            phlower_tensor(
                phlower_array(
                    sp.random(
                        shape[0], shape[1], density=0.1, random_state=rng
                    ),
                ).to_tensor(),
                dimension=PhysicalDimensions(dims),
            )
            for shape, dims in zip(shapes, dimensions, strict=True)
        ]

    return _create


@pytest.fixture
def create_dense_tensors() -> (
    Callable[
        [list[tuple[int]], list[dict[str, float]] | None, bool],
        list[IPhlowerTensor],
    ]
):
    def _create(
        shapes: list[tuple],
        dimensions: list[dict] | None = None,
        is_time_series: bool = False,
    ) -> list[IPhlowerTensor]:
        if dimensions is None:
            return [
                phlower_tensor(
                    np.random.rand(*shape),
                    is_time_series=is_time_series,
                    dtype=torch.float32,
                )
                for shape in shapes
            ]

        return [
            phlower_tensor(
                np.random.rand(*shape),
                dimension=dims,
                is_time_series=is_time_series,
                dtype=torch.float32,
            )
            for shape, dims in zip(shapes, dimensions, strict=True)
        ]

    return _create
