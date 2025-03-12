import pathlib
import random
import shutil

import numpy as np
import pytest
import scipy.sparse as sp
from phlower.services.preprocessing import PhlowerScalingService

OUTPUT_DIR = pathlib.Path(__file__).parent / "_out"
DATA_DIR = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def prepare_sample_interim_files() -> list[pathlib.Path]:
    output_base_dir = OUTPUT_DIR
    if output_base_dir.exists():
        shutil.rmtree(output_base_dir)

    random.seed(11)
    np.random.seed(11)
    base_interim_dir = output_base_dir / "interim"
    base_interim_dir.mkdir(parents=True)

    interim_dirs: list[pathlib.Path] = []
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

        rng = np.random.default_rng()
        for sparse_name in [
            "nodal_adj",
            "nodal_x_grad_hop1",
            "nodal_y_grad_hop1",
            "nodal_z_grad_hop1",
        ]:
            sparse = sp.random(n_nodes, n_nodes, density=0.1, random_state=rng)
            sp.save_npz(interim_dir / sparse_name, sparse.tocoo().astype(dtype))
        (interim_dir / "converted").touch()
        interim_dirs.append(interim_dir)

    return interim_dirs


def generate_interim_data() -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}

    n_nodes = 200
    nodal_initial_u = np.random.rand(n_nodes, 3, 1)
    nodal_last_u = np.random.rand(n_nodes, 3, 1)
    results.update(
        {"nodal_initial_u": nodal_initial_u, "nodal_last_u": nodal_last_u}
    )

    rng = np.random.default_rng()
    for sparse_name in [
        "nodal_adj",
        "nodal_x_grad_hop1",
        "nodal_y_grad_hop1",
        "nodal_z_grad_hop1",
    ]:
        sparse = sp.random(n_nodes, n_nodes, density=0.1, random_state=rng)
        results.update({sparse_name: sparse})

    return results


@pytest.mark.parametrize("yaml_file", ["preprocess.yml"])
def test__scaling_with_lazy_loading(
    yaml_file: str, prepare_sample_interim_files: list[pathlib.Path]
):
    scaler = PhlowerScalingService.from_yaml(DATA_DIR / yaml_file)
    scaler.fit_transform_all(
        interim_data_directories=prepare_sample_interim_files,
        output_base_directory=OUTPUT_DIR / "preprocessed",
    )

    # check corresponding cases are existed.
    case_names = [p.name for p in prepare_sample_interim_files]
    for case_name in case_names:
        assert (OUTPUT_DIR / f"preprocessed/{case_name}").exists()


@pytest.mark.parametrize("yaml_file", ["preprocess.yml"])
def test__scaling_with_OnMemory_data(
    yaml_file: str, prepare_sample_interim_files: list[pathlib.Path]
):
    directory_name = "preprocessed_2"
    scaler = PhlowerScalingService.from_yaml(DATA_DIR / yaml_file)
    scaler.fit_transform_all(
        interim_data_directories=prepare_sample_interim_files,
        output_base_directory=OUTPUT_DIR / directory_name,
    )
    scaler.save(OUTPUT_DIR / directory_name, "scaled")

    new_scaler = PhlowerScalingService.from_yaml(
        OUTPUT_DIR / f"{directory_name}/scaled.yml"
    )

    targets = generate_interim_data()
    results = new_scaler.transform(targets)

    # check whether interim variable name is existed.
    for name in targets.keys():
        assert name in results
