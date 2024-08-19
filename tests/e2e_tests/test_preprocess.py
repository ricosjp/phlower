import pathlib
import shutil

import numpy as np
import pytest
import scipy.sparse as sp
from phlower.io import PhlowerDirectory, PhlowerFileBuilder
from phlower.services.preprocessing import PhlowerScalingService
from phlower.settings import PhlowerScalingSetting, PhlowerSetting
from pipe import select, where

_OUTPUT_DIR = pathlib.Path(__file__).parent / "_tmp_preprocess"


@pytest.fixture(scope="module")
def prepare_sample_interim_files():
    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR)
    _OUTPUT_DIR.mkdir()

    base_preprocessed_dir = _OUTPUT_DIR / "interim"
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

        sparse_array_names = [
            "nodal_adj",
            "nodal_x_grad_hop1",
            "nodal_y_grad_hop1",
            "nodal_z_grad_hop1",
        ]
        rng = np.random.default_rng()
        for name in sparse_array_names:
            arr = sp.random(n_nodes, n_nodes, density=0.1, random_state=rng)
            sp.save_npz(preprocessed_dir / name, arr.tocoo().astype(dtype))

        (preprocessed_dir / "converted").touch()


@pytest.fixture(scope="module")
def perform_scaling(prepare_sample_interim_files: None) -> PhlowerSetting:
    path = _OUTPUT_DIR
    phlower_path = PhlowerDirectory(path)

    interim_directories = list(
        phlower_path.find_directory(
            required_filename="converted", recursive=True
        )
    )
    output_base_directory = _OUTPUT_DIR / "preprocessed"

    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/preprocess.yml")

    scaler = PhlowerScalingService.from_setting(setting)
    scaler.fit_transform_all(
        interim_data_directories=interim_directories,
        output_base_directory=output_base_directory,
    )

    return setting


@pytest.mark.e2e_test
@pytest.mark.parametrize(
    "interim_base_directory, scaling_base_directory",
    [(_OUTPUT_DIR / "interim", _OUTPUT_DIR / "preprocessed")],
)
def test__saved_array_is_same_as_saved_scalers_transformed(
    interim_base_directory: pathlib.Path,
    scaling_base_directory: pathlib.Path,
    perform_scaling: None,
):
    phlower_path = PhlowerDirectory(interim_base_directory)
    interim_directories = list(
        phlower_path.find_directory(
            required_filename="converted", recursive=True
        )
    )

    variable_names: list[str] = list(
        interim_directories[0].iterdir()
        | where(lambda x: x.name.endswith(".npy"))
        | select(lambda x: PhlowerFileBuilder.numpy_file(x))
        | select(lambda x: x.get_variable_name())
    )
    assert len(variable_names) > 0

    saved_setting = PhlowerScalingSetting.read_yaml(
        scaling_base_directory / "preprocess.yml"
    )
    restored_scaler = PhlowerScalingService(saved_setting)

    for interim_dir in interim_directories:
        path = PhlowerDirectory(interim_dir)
        saved_path = PhlowerDirectory(
            _OUTPUT_DIR / f"preprocessed/{interim_dir.name}"
        )
        for name in variable_names:
            file_path = path.find_variable_file(name)
            transformed = restored_scaler.transform_file(name, file_path)

            saved_file_path = saved_path.find_variable_file(name)
            saved_arr = saved_file_path.load()

            if saved_arr.is_sparse:
                np.testing.assert_array_almost_equal(
                    transformed.todense(), saved_arr.to_numpy().todense()
                )
            else:
                np.testing.assert_array_almost_equal(
                    transformed, saved_arr.to_numpy()
                )
