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
from phlower_tensor.collections import phlower_tensor_collection

from phlower.utils._extended_simulation_field import (
    FieldDataOverwriteContext,
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

# region test for FieldDataOverrideContext


def test__field_data_override_context():
    rng = np.random.default_rng(seed=11)

    v1_1 = rng.random((10, 3)).astype(np.float32)
    v1_2 = rng.random((10, 3)).astype(np.float32)

    # Ensure that the two arrays are not equal
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(v1_1, v1_2)

    source = phlower_tensor_collection({"v1": phlower_tensor(v1_1)})
    field = create_simulation_field(
        field_tensors={
            "v1": phlower_tensor(v1_2),
        },
        batch_info={},
    )
    replacement_keys = ["v1"]

    with FieldDataOverwriteContext(
        source, field, replacement_keys=replacement_keys
    ) as context:
        for k in replacement_keys:
            assert k not in context.source
            np.testing.assert_array_almost_equal(
                context.field_data[k].to_tensor().numpy(), v1_1
            )

    # After exiting the context, the original data should be restored
    for k in replacement_keys:
        assert k in source
        np.testing.assert_array_almost_equal(
            source[k].to_tensor().numpy(), v1_1
        )
        np.testing.assert_array_almost_equal(field[k].to_tensor().numpy(), v1_2)


def test__context_when_field_data_is_missing():
    source = phlower_tensor_collection(
        {"v1": phlower_tensor(torch.rand(10, 3))}
    )
    field = None
    replacement_keys = ["v1"]

    with pytest.raises(
        ValueError,
        match="Field data is None, but replacement keys are provided.",
    ):
        with FieldDataOverwriteContext(
            source, field, replacement_keys=replacement_keys
        ):
            pass


def test__context_when_shape_mismatch():
    rng = np.random.default_rng(seed=11)

    v1_1 = rng.random((10, 3)).astype(np.float32)
    v1_2 = rng.random((5, 3)).astype(np.float32)  # Different shape

    source = phlower_tensor_collection({"v1": phlower_tensor(v1_1)})
    field = create_simulation_field(
        field_tensors={
            "v1": phlower_tensor(v1_2),
        },
        batch_info={},
    )
    replacement_keys = ["v1"]

    with pytest.raises(
        ValueError,
        match="Replacement data must have the same shape",
    ):
        with FieldDataOverwriteContext(
            source, field, replacement_keys=replacement_keys
        ):
            pass


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Not supported in Python versions below 3.12",
)
def test__field_data_override_of_graphlow_simulation_field():
    rng = np.random.default_rng(seed=11)

    v1_1 = rng.random((10, 3)).astype(np.float32)
    v1_2 = rng.random((10, 3)).astype(np.float32)

    # Ensure that the two arrays are not equal
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(v1_1, v1_2)

    source = phlower_tensor_collection({"v1": phlower_tensor(v1_1)})
    pv_mesh = pv.ImageData(dimensions=(10, 10, 10), spacing=(1.0, 1.0, 1.0))
    pv_mesh.point_data["v2"] = np.random.rand(pv_mesh.n_points, 3).astype(
        np.float32
    )
    tensor_mesh = PyVistaMeshAdapter(
        pv_mesh=pv_mesh,
        dimensions={},
    )

    field = create_simulation_field(
        field_tensors={
            "v1": phlower_tensor(v1_2),
        },
        batch_info={},
        tensor_mesh=tensor_mesh,
    )
    replacement_keys = ["v1"]

    with FieldDataOverwriteContext(
        source, field, replacement_keys=replacement_keys
    ) as context:
        for k in replacement_keys:
            assert k not in context.source
            np.testing.assert_array_almost_equal(
                context.field_data[k].to_tensor().numpy(), v1_1
            )

    # After exiting the context, the original data should be restored
    for k in replacement_keys:
        assert k in source
        np.testing.assert_array_almost_equal(
            source[k].to_tensor().numpy(), v1_1
        )
        np.testing.assert_array_almost_equal(field[k].to_tensor().numpy(), v1_2)


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Not supported in Python versions below 3.12",
)
@pytest.mark.parametrize(
    "replacement_keys",
    [
        ["v1", "v2"],
        ["v1", "v2", "v3"],
    ],
)
def test__not_allowed_to_overwrite_point_or_cell_data(
    replacement_keys: list[str],
):
    rng = np.random.default_rng(seed=11)
    source = phlower_tensor_collection(
        {
            key: phlower_tensor(rng.random((10, 3)).astype(np.float32))
            for key in replacement_keys
        }
    )

    pv_mesh = pv.ImageData(dimensions=(10, 10, 10), spacing=(1.0, 1.0, 1.0))
    pv_mesh.point_data["v2"] = np.random.rand(pv_mesh.n_points, 3).astype(
        np.float32
    )
    pv_mesh.cell_data["v3"] = np.random.rand(pv_mesh.n_cells, 3).astype(
        np.float32
    )
    tensor_mesh = PyVistaMeshAdapter(
        pv_mesh=pv_mesh,
        dimensions={},
    )

    field = create_simulation_field(
        field_tensors={
            "v1": phlower_tensor(rng.random((10, 3)).astype(np.float32)),
        },
        batch_info={},
        tensor_mesh=tensor_mesh,
    )

    with pytest.raises(
        NotImplementedError, match="Overwriting mesh data is not supported."
    ):
        with FieldDataOverwriteContext(
            source, field, replacement_keys=replacement_keys
        ):
            pass


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="Not supported in Python versions below 3.12",
)
def test__context_when_shape_mismatch_for_graphlow_mesh():
    rng = np.random.default_rng(seed=11)

    v1_1 = rng.random((10, 3)).astype(np.float32)
    v1_2 = rng.random((5, 3)).astype(np.float32)  # Different shape

    source = phlower_tensor_collection({"v1": phlower_tensor(v1_1)})

    pv_mesh = pv.ImageData(dimensions=(10, 10, 10), spacing=(1.0, 1.0, 1.0))
    pv_mesh.point_data["v2"] = np.random.rand(pv_mesh.n_points, 3).astype(
        np.float32
    )
    pv_mesh.cell_data["v3"] = np.random.rand(pv_mesh.n_cells, 3).astype(
        np.float32
    )
    tensor_mesh = PyVistaMeshAdapter(
        pv_mesh=pv_mesh,
        dimensions={},
    )
    field = create_simulation_field(
        field_tensors={
            "v1": phlower_tensor(v1_2),
        },
        batch_info={},
        tensor_mesh=tensor_mesh,
    )
    replacement_keys = ["v1"]

    with pytest.raises(
        ValueError,
        match="Replacement data must have the same shape",
    ):
        with FieldDataOverwriteContext(
            source, field, replacement_keys=replacement_keys
        ):
            pass


# endregion
