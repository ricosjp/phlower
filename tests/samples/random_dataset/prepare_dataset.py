import pathlib
import random
import shutil

import numpy as np
import scipy.sparse as sp


def prepare_sample_interim_files(output_directory: pathlib.Path):
    random.seed(0)
    np.random.seed(0)

    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir()

    base_interim_dir = output_directory / "interim"
    base_interim_dir.mkdir()

    n_cases = 3
    dtype = np.float32
    for i in range(n_cases):
        n_nodes = 100 * (i + 1)
        interim_dir = base_interim_dir / f"case_{i}"
        interim_dir.mkdir()

        nodal_initial_u = np.random.rand(n_nodes, 3, 1)
        np.save(
            interim_dir / "nodal_initial_u.npy",
            nodal_initial_u.astype(dtype),
        )

        # nodal_last_u = np.random.rand(n_nodes, 3, 1)
        np.save(interim_dir / "nodal_last_u.npy", nodal_initial_u.astype(dtype))

        sparse_array_names = [
            "nodal_adj",
            "nodal_x_grad_hop1",
            "nodal_y_grad_hop1",
            "nodal_z_grad_hop1",
        ]
        rng = np.random.default_rng()
        for name in sparse_array_names:
            arr = sp.random(n_nodes, n_nodes, density=0.1, random_state=rng)
            sp.save_npz(interim_dir / name, arr.tocoo().astype(dtype))

        (interim_dir / "converted").touch()


if __name__ == "__main__":
    output_directory = pathlib.Path(__file__).parent / "out"
    prepare_sample_interim_files(output_directory)
