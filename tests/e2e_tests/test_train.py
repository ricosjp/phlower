import pathlib
import random
import shutil

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from phlower.io import PhlowerDirectory
from phlower.services.trainer import PhlowerTrainer
from phlower.settings import PhlowerSetting

_OUTPUT_DIR = pathlib.Path(__file__).parent / "_tmp"

random.seed(11)
np.random.seed(11)
torch.manual_seed(11)


@pytest.fixture(scope="module")
def prepare_sample_preprocessed_files():
    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR)
    _OUTPUT_DIR.mkdir()

    base_preprocessed_dir = _OUTPUT_DIR / "preprocessed"
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


def test__simple_training(prepare_sample_preprocessed_files):
    path = _OUTPUT_DIR
    phlower_path = PhlowerDirectory(path)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/setting.yml")
    setting.model.resolve(is_first=True)

    trainer = PhlowerTrainer(setting)

    model, loss = trainer.train(
        preprocessed_directories=preprocessed_directories, n_epoch=2
    )
    model.draw(_OUTPUT_DIR)

    assert loss.has_dimension
    assert not torch.isinf(loss.to_tensor())
    assert not torch.isnan(loss.to_tensor())

