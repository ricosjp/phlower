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

        np.save(
            interim_dir / "nodal_coordinates.npy",
            np.random.rand(n_nodes, 3).astype(dtype),
        )

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


def generate_interim_data(
    dense_interim_names: list[str], sparse_interim_names: list[str]
) -> dict[str, np.ndarray]:
    results: dict[str, np.ndarray] = {}

    n_nodes = 200
    results.update(
        {name: np.random.rand(n_nodes, 3, 1) for name in dense_interim_names}
    )

    rng = np.random.default_rng()
    for sparse_name in sparse_interim_names:
        sparse = sp.random(n_nodes, n_nodes, density=0.1, random_state=rng)
        results.update({sparse_name: sparse})

    return results


@pytest.mark.parametrize(
    "yaml_file", ["preprocess.yml", "preprocess_with_physics.yml"]
)
def test__scaling_with_lazy_loading(
    yaml_file: str,
    prepare_sample_interim_files: list[pathlib.Path],
    tmp_path: pathlib.Path,
):
    scaler = PhlowerScalingService.from_yaml(DATA_DIR / yaml_file)
    scaler.fit_transform_all(
        interim_data_directories=prepare_sample_interim_files,
        output_base_directory=tmp_path / "preprocessed",
    )

    # check corresponding cases are existed.
    case_names = [p.name for p in prepare_sample_interim_files]
    for case_name in case_names:
        assert (tmp_path / f"preprocessed/{case_name}").exists()


@pytest.mark.parametrize(
    "yaml_file, dense_names, sparse_names",
    [
        (
            "preprocess.yml",
            ["nodal_initial_u", "nodal_last_u"],
            [
                "nodal_adj",
                "nodal_x_grad_hop1",
                "nodal_y_grad_hop1",
                "nodal_z_grad_hop1",
            ],
        ),
        (
            "preprocess_with_physics.yml",
            ["nodal_initial_u", "nodal_last_u", "nodal_coordinates"],
            [],
        ),
    ],
)
def test__scaling_with_OnMemory_data(
    yaml_file: str,
    dense_names: list[str],
    sparse_names: list[str],
    prepare_sample_interim_files: list[pathlib.Path],
    tmp_path: pathlib.Path,
):
    scaler = PhlowerScalingService.from_yaml(DATA_DIR / yaml_file)
    scaler.fit_transform_all(
        interim_data_directories=prepare_sample_interim_files,
        output_base_directory=tmp_path / "preprocessed",
    )

    new_scaler = PhlowerScalingService.from_yaml(
        tmp_path / "preprocessed/preprocess.yml"
    )

    targets = generate_interim_data(
        dense_interim_names=dense_names, sparse_interim_names=sparse_names
    )
    results = new_scaler.transform(targets)

    # check whether interim variable name is existed.
    for name in dense_names + sparse_names:
        assert name in results


@pytest.mark.parametrize(
    "yaml_file",
    ["preprocess.yml", "preprocess_2.yml", "preprocess_with_physics.yml"],
)
def test__retrieve_scaling_items_from_dumped_data(
    yaml_file: str,
    prepare_sample_interim_files: list[pathlib.Path],
    tmp_path: pathlib.Path,
):
    scaler = PhlowerScalingService.from_yaml(DATA_DIR / yaml_file)
    scaler.fit_transform_all(
        interim_data_directories=prepare_sample_interim_files,
        output_base_directory=tmp_path,
    )
    scaler.save(tmp_path, "scaled")
    dumped = scaler._recreate_setting().model_dump()

    new_scaler = PhlowerScalingService.from_yaml(tmp_path / "scaled.yml")
    loaded = new_scaler._recreate_setting().model_dump()

    assert dumped == loaded
