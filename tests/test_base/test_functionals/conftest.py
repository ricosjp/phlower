from collections.abc import Callable

import numpy as np
import pytest
from phlower._base import PhysicalDimensions
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
                phlower_array(
                    sp.random(
                        shape[0], shape[1], density=0.1, random_state=rng
                    ),
                ).to_phlower_tensor()
                for shape in shapes
            ]

        return [
            phlower_array(
                sp.random(shape[0], shape[1], density=0.1, random_state=rng),
                dimensions=PhysicalDimensions(dims),
            ).to_phlower_tensor()
            for shape, dims in zip(shapes, dimensions, strict=True)
        ]

    return _create


@pytest.fixture
def create_dense_tensors() -> (
    Callable[
        [list[tuple[int]], list[dict[str, float]] | None], list[IPhlowerTensor]
    ]
):
    def _create(
        shapes: list[tuple], dimensions: list[dict] | None = None
    ) -> list[IPhlowerTensor]:
        if dimensions is None:
            return [
                phlower_array(np.random.rand(*shape)).to_phlower_tensor()
                for shape in shapes
            ]

        return [
            phlower_array(
                np.random.rand(*shape), dimensions=dims
            ).to_phlower_tensor()
            for shape, dims in zip(shapes, dimensions, strict=True)
        ]

    return _create
