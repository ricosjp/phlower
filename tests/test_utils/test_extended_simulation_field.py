import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import pyvista as pv
import torch
from phlower_tensor import (
    PhlowerTensor,
    PhysicalDimensions,
    SimulationField,
    phlower_tensor,
)

from phlower.utils._extended_simulation_field import (
    GraphlowSimulationField,
    PyVistaMeshAdapter,
    create_simulation_field,
)
from phlower.utils._lazy_import import _LazyImport

if TYPE_CHECKING:
    import graphlow
else:
    # NOTE: From Python 3.15, we can use `lazy import` directly.
    # For now, we use a custom lazy import mechanism.
    graphlow = _LazyImport("graphlow")


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Not supported in Python versions below 3.12",
)
def test__create_simulation_field():
    field_tensors = {
        "v1": phlower_tensor(torch.rand(10, 3, dtype=torch.float32)),
        "s": phlower_tensor(torch.rand(10, 1, dtype=torch.float32)),
    }
    field = create_simulation_field(
        field_tensors=field_tensors,
        batch_info={},
    )
    assert isinstance(field, SimulationField)

    tensor_mesh = PyVistaMeshAdapter(
        pv.ImageData(dimensions=(10, 1, 1), spacing=(1.0, 1.0, 1.0)),
        dimensions={"v2": {"L": 1, "T": -1}},
    )

    field = create_simulation_field(
        field_tensors=field_tensors, batch_info={}, tensor_mesh=tensor_mesh
    )
    assert isinstance(field, GraphlowSimulationField)


# region test for GraphlowSimulationField


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Not supported in Python versions below 3.12",
)
def test__dimensions_of_graphlow_simulation_field():
    desired_dimensins = {
        "v1": {"L": 1, "T": -1},
        "p": {"M": 1, "T": -2, "L": -1},
        "v2": {"L": 1, "T": -1},
    }

    field_tensors = {
        "v1": phlower_tensor(
            torch.rand(10, 3, dtype=torch.float32),
            dimension=desired_dimensins["v1"],
        ),
        "p": phlower_tensor(
            torch.rand(10, 1, dtype=torch.float32),
            dimension=desired_dimensins["p"],
        ),
    }
    pv_mesh = pv.ImageData(dimensions=(10, 10, 10), spacing=(1.0, 1.0, 1.0))
    pv_mesh.point_data["v2"] = np.random.rand(pv_mesh.n_points, 3).astype(
        np.float32
    )

    tensor_mesh = PyVistaMeshAdapter(
        pv_mesh=pv_mesh,
        dimensions=desired_dimensins,
    )
    field = create_simulation_field(
        field_tensors=field_tensors, batch_info={}, tensor_mesh=tensor_mesh
    )

    for k, v in desired_dimensins.items():
        assert k in field.keys()
        assert k in field

        actual = field[k]
        assert isinstance(actual, PhlowerTensor)
        assert actual.dimension.to_physics_dimension() == PhysicalDimensions(v)


# endregion
