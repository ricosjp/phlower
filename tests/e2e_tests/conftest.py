import pathlib
import random
import shutil

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture(scope="module")
def prepare_sample_preprocessed_files_fixture() -> pathlib.Path:
    output_base_directory = pathlib.Path(__file__).parent / "_tmp"

    random.seed(11)
    np.random.seed(11)
    if output_base_directory.exists():
        shutil.rmtree(output_base_directory)
    output_base_directory.mkdir()

    base_preprocessed_dir = output_base_directory / "preprocessed"
    base_preprocessed_dir.mkdir()

    n_cases = 3
    dtype = np.float32
    for i in range(n_cases):
        n_nodes = 100 * (i + 1)
        preprocessed_dir = base_preprocessed_dir / f"case_{i}"
        preprocessed_dir.mkdir()

        nodal_initial_u = np.random.rand(n_nodes, 3, 1)
        np.save(
            preprocessed_dir / "nodal_initial_u.npy",
            nodal_initial_u.astype(dtype),
        )

        # nodal_last_u = np.random.rand(n_nodes, 3, 1)
        np.save(
            preprocessed_dir / "nodal_last_u.npy", nodal_initial_u.astype(dtype)
        )

        rng = np.random.default_rng()
        nodal_nadj = sp.random(n_nodes, n_nodes, density=0.1, random_state=rng)
        sp.save_npz(
            preprocessed_dir / "nodal_nadj", nodal_nadj.tocoo().astype(dtype)
        )

        (preprocessed_dir / "preprocessed").touch()

    return output_base_directory
