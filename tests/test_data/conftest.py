import pathlib
import secrets
import shutil

import numpy as np
import pytest
import pyvista as pv
import scipy.sparse as sp

from phlower.io import PhlowerNumpyFile
from phlower.utils.typing import ArrayDataType

_OUTPUT_BASE_DIRECTORY = pathlib.Path(__file__).parent / "tmp/datasets"
_OUTPUT_BASE_DIRECTORY_W_MESH = (
    pathlib.Path(__file__).parent / "tmp/datasets_with_mesh"
)


@pytest.fixture
def output_base_directory() -> pathlib.Path:
    return _OUTPUT_BASE_DIRECTORY


@pytest.fixture
def output_base_directory_w_mesh() -> pathlib.Path:
    return _OUTPUT_BASE_DIRECTORY_W_MESH


def create_dataset(
    output_base_directory: pathlib.Path,
    directory_names: list[str],
    name2dense_shape: dict[str, tuple[int, ...]],
    name2sparse_shape: dict[str, tuple[int, ...]],
    name2field_shape: dict[str, tuple[int, ...]] | None = None,
    include_mesh: bool = False,
    mesh_n_points: int = 20,
    encrypt_key: bytes | None = None,
) -> None:
    if output_base_directory.exists():
        shutil.rmtree(output_base_directory)
    output_base_directory.mkdir(parents=True)

    rng = np.random.default_rng(seed=0)
    for name in directory_names:
        _output_directory = output_base_directory / name
        _output_directory.mkdir()

        for v_name, v_shape in name2dense_shape.items():
            arr = rng.random(v_shape)
            PhlowerNumpyFile.save(
                _output_directory, v_name, arr, encrypt_key=encrypt_key
            )

        for v_name, v_shape in name2sparse_shape.items():
            arr = sp.random(*v_shape, density=0.1)
            PhlowerNumpyFile.save(
                _output_directory, v_name, arr, encrypt_key=encrypt_key
            )

        if include_mesh and name2field_shape is not None:
            mesh = pv.ImageData(
                dimensions=(mesh_n_points, 1, 1),
                origin=(0, 0, 0),
                spacing=(1.0, 1.0, 1.0),
            )
            for v_name, v_shape in name2field_shape.items():
                mesh.point_data[v_name] = rng.random(v_shape, dtype=np.float32)
            mesh_path = _output_directory / "mesh.vtk"
            mesh.save(mesh_path)


@pytest.fixture(scope="module")
def create_base_dataset() -> dict[str, dict[str, ArrayDataType]]:

    name2dense_shape: dict[str, tuple[int, ...]] = {
        "x0": (1, 3, 4, 1),
        "x1": (10, 5, 1),
        "x2": (11, 3, 1),
        "y0": (1, 3, 4, 1),
    }
    name2sparse_shape: dict[str, tuple[int, ...]] = {
        "s0": (5, 5),
        "s1": (10, 5),
    }

    create_dataset(
        output_base_directory=_OUTPUT_BASE_DIRECTORY,
        directory_names=["data0", "data1", "data2"],
        name2dense_shape=name2dense_shape,
        name2sparse_shape=name2sparse_shape,
        include_mesh=False,
        encrypt_key=None,
    )


@pytest.fixture(scope="module")
def create_dataset_with_mesh():
    if _OUTPUT_BASE_DIRECTORY_W_MESH.exists():
        shutil.rmtree(_OUTPUT_BASE_DIRECTORY_W_MESH)
    _OUTPUT_BASE_DIRECTORY_W_MESH.mkdir(parents=True)

    directory_names = ["data0", "data1", "data2"]
    name2dense_shape: dict[str, tuple[int, ...]] = {
        "x0": (1, 3, 4, 1),
        "x1": (10, 5, 1),
        "x2": (11, 3, 1),
        "y0": (1, 3, 4, 1),
    }
    name2sparse_shape: dict[str, tuple[int, ...]] = {
        "s0": (5, 5),
        "s1": (10, 5),
    }
    name2field_shape: dict[str, tuple[int, ...]] = {
        "velocity1": (20, 3),
        "pressure1": (20, 1),
    }

    create_dataset(
        output_base_directory=_OUTPUT_BASE_DIRECTORY_W_MESH,
        directory_names=directory_names,
        name2dense_shape=name2dense_shape,
        name2sparse_shape=name2sparse_shape,
        name2field_shape=name2field_shape,
        include_mesh=True,
        encrypt_key=None,
    )


@pytest.fixture(scope="module")
def create_encrypted_dataset() -> bytes:
    directory_names = ["data0", "data1", "data2"]
    name2dense_shape: dict[str, tuple[int, ...]] = {
        "x0": (1, 3, 4),
        "x1": (10, 5),
        "x2": (11, 3, 1),
        "x3": (11, 3),
        "y0": (1, 3, 4, 1),
        "y1": (1, 3, 4),
    }
    name2sparse_shape: dict[str, tuple[int, ...]] = {
        "s0": (5, 5),
        "s1": (10, 5),
    }
    name2field_shape: dict[str, tuple[int, ...]] = {
        "v0": (20, 3),
        "v1": (20, 1),
    }

    token = secrets.token_bytes(32)
    create_dataset(
        output_base_directory=_OUTPUT_BASE_DIRECTORY,
        directory_names=directory_names,
        name2dense_shape=name2dense_shape,
        name2sparse_shape=name2sparse_shape,
        name2field_shape=name2field_shape,
        include_mesh=True,
        mesh_n_points=20,
        encrypt_key=token,
    )

    return token
