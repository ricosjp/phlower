from collections.abc import Callable

import numpy as np
import pytest
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
        shapes: list[tuple], dimensions: list[dict] = None
    ) -> list[IPhlowerTensor]:
        rng = np.random.default_rng()
        arrs = [
            phlower_array(
                sp.random(shape[0], shape[1], density=0.1, random_state=rng)
            )
            for shape in shapes
        ]
        if dimensions is None:
            return [arr.to_phlower_tensor().coalesce() for arr in arrs]
        else:
            return [
                arr.to_phlower_tensor(dimension=dimensions[i]).coalesce()
                for i, arr in enumerate(arrs)
            ]

    return _create


@pytest.fixture
def create_dense_tensors() -> (
    Callable[
        [list[tuple[int]], list[dict[str, float]] | None], list[IPhlowerTensor]
    ]
):
    def _create(
        shapes: list[tuple], dimensions: list[dict] = None
    ) -> list[IPhlowerTensor]:
        arrs = [phlower_array(np.random.rand(*shape)) for shape in shapes]
        if dimensions is None:
            return [arr.to_phlower_tensor() for arr in arrs]
        else:
            return [
                arr.to_phlower_tensor(dimension=dimensions[i])
                for i, arr in enumerate(arrs)
            ]

    return _create
