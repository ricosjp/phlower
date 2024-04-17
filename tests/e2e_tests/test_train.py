import pathlib
import random
import shutil

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from phlower.collections.tensors import phlower_tensor_collection
from phlower.data import DataLoaderBuilder, LazyPhlowerDataset
from phlower.io import PhlowerDirectory
from phlower.nn import GCN
from phlower.services.loss_operations import LossCalculator
from phlower.settings import PhlowerTrainerSetting

_OUTPUT_DIR = pathlib.Path(__file__).parent / "_tmp"


@pytest.fixture(scope="module")
def prepare_sample_preprocessed_files():
    random.seed(11)
    np.random.seed(11)

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

        nodal_last_u = np.random.rand(n_nodes, 3, 1)
        np.save(
            preprocessed_dir / "nodal_last_u.npy", nodal_last_u.astype(dtype)
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

    targets = [
        p
        for p in phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    ]

    dataset = LazyPhlowerDataset(
        x_variable_names=["nodal_initial_u"],
        y_variable_names=["nodal_last_u"],
        support_names=["nodal_nadj"],
        directories=targets,
    )

    setting = PhlowerTrainerSetting(
        variable_dimensions={
            "nodal_initial_u": {"length": 1, "time": -1},
            "nodal_last_u": {"length": 1, "time": -1},
            "nodal_nadj": {},
        }
    )
    builder = DataLoaderBuilder(setting)

    data_loader = builder.create(dataset, shuffle=True)

    # loss_function = torch.nn.functional.mse_loss
    loss_function = LossCalculator.from_dict(name2loss={"nodal_last_u": "mse"})
    model = GCN(
        nodes=[1, 16, 1], input_key="nodal_initial_u", support_name="nodal_nadj"
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    counter = 0
    for batch in data_loader:
        optimizer.zero_grad()

        h = model.forward(batch.x_data, supports=batch.sparse_supports)

        dict_data = phlower_tensor_collection({"nodal_last_u": h})
        losses = loss_function.calculate(dict_data, batch.y_data)
        loss = loss_function.aggregate(losses)
        loss.backward()

        optimizer.step()

        assert loss.has_dimension
        counter += 1

    # confirm that codes in the loop has finished
    assert counter == len(dataset)
