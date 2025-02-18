import pathlib
import random
import shutil
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch
import yaml
from phlower import PhlowerTensor
from phlower.io import PhlowerDirectory, select_snapshot_file
from phlower.services.trainer import PhlowerTrainer
from phlower.settings import PhlowerSetting
from phlower.utils.exceptions import PhlowerRestartTrainingCompletedError

_OUTPUT_DIR = pathlib.Path(__file__).parent / "_out"
_SETTINGS_DIR = pathlib.Path(__file__).parent / "data"


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


@pytest.fixture(scope="module")
def simple_training(prepare_sample_preprocessed_files: None) -> PhlowerTensor:
    phlower_path = PhlowerDirectory(_OUTPUT_DIR)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(_SETTINGS_DIR / "train.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_DIR / "model"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    loss = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )
    return loss


def test__simple_training_without_validation_dataset(
    prepare_sample_preprocessed_files: None,
):
    phlower_path = PhlowerDirectory(_OUTPUT_DIR)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(_SETTINGS_DIR / "train.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_DIR / "model_wo_validation"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    _ = trainer.train(
        train_directories=preprocessed_directories,
        output_directory=output_directory,
    )


def test__training_with_multiple_batch_size(
    prepare_sample_preprocessed_files: None,
):
    phlower_path = PhlowerDirectory(_OUTPUT_DIR)
    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(_SETTINGS_DIR / "train_batch_size.yml")
    assert setting.training.batch_size > 1

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_DIR / "model_batch_size"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    loss = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )
    assert loss.has_dimension
    assert not torch.isinf(loss.to_tensor())
    assert not torch.isnan(loss.to_tensor())


def test__simple_training(simple_training: PhlowerTensor):
    loss: PhlowerTensor = simple_training

    assert loss.has_dimension
    assert not torch.isinf(loss.to_tensor())
    assert not torch.isnan(loss.to_tensor())


@pytest.fixture
def perform_restart() -> Callable[[int | None], None]:
    def restart_training(n_epoch: int | None = None):
        with open(_SETTINGS_DIR / "train.yml") as fr:
            content = yaml.load(fr, Loader=yaml.SafeLoader)

        if n_epoch is not None:
            # NOTE: overwrite n_epoch to restart
            content["training"]["n_epoch"] = n_epoch

        setting = PhlowerSetting(**content)
        trainer = PhlowerTrainer.from_setting(setting)

        restart_directory = _OUTPUT_DIR / "model"
        trainer._reinit_for_restart(restart_directory=restart_directory)

        phlower_path = PhlowerDirectory(_OUTPUT_DIR)
        preprocessed_directories = list(
            phlower_path.find_directory(
                required_filename="preprocessed", recursive=True
            )
        )
        trainer.train(
            output_directory=restart_directory,
            train_directories=preprocessed_directories,
            validation_directories=preprocessed_directories,
        )

    return restart_training


def test__not_allowed_restart_when_all_epoch_is_finished(
    simple_training: PhlowerTensor,
    perform_restart: Callable[[int | None], None],
):
    with pytest.raises(PhlowerRestartTrainingCompletedError):
        perform_restart()


def test__last_epoch_is_update_after_restart(
    simple_training: PhlowerTensor,
    perform_restart: Callable[[int | None], None],
):
    last_snapshot = select_snapshot_file(_OUTPUT_DIR / "model", "latest")
    assert last_snapshot.file_path.name.startswith("snapshot_epoch_9")

    n_epoch = 12
    perform_restart(n_epoch)

    last_snapshot = select_snapshot_file(_OUTPUT_DIR / "model", "latest")
    assert last_snapshot.file_path.name.startswith(
        f"snapshot_epoch_{n_epoch - 1}"
    )

    # check log.csv
    df = pd.read_csv(
        _OUTPUT_DIR / "model/log.csv",
        header=0,
        index_col=None,
        skipinitialspace=True,
    )

    assert max(df.loc[:, "epoch"]) == n_epoch - 1

    # check elapsed_time increases monotonically
    _prev = 0.0
    for v in df.loc[:, "elapsed_time"]:
        assert v > _prev
        _prev = v


@pytest.fixture
def clean_directories():
    output_directory = _OUTPUT_DIR / "base_model"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    output_directory = _OUTPUT_DIR / "pretrained_model"
    if output_directory.exists():
        shutil.rmtree(output_directory)


def test__starting_pretrained_state(clean_directories: None):
    # base model
    setting = PhlowerSetting.read_yaml(_SETTINGS_DIR / "train.yml")
    trainer = PhlowerTrainer.from_setting(setting)

    phlower_path = PhlowerDirectory(_OUTPUT_DIR)
    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )
    trainer.train(
        output_directory=(_OUTPUT_DIR / "base_model"),
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
    )

    # start from pretrained state
    trainer = PhlowerTrainer.from_setting(setting)
    trainer.load_pretrained(_OUTPUT_DIR / "base_model", "best")
    trainer.train(
        output_directory=(_OUTPUT_DIR / "pretrained_model"),
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
    )

    # check log.csv
    df = pd.read_csv(
        _OUTPUT_DIR / "base_model/log.csv",
        header=0,
        index_col=None,
        skipinitialspace=True,
    )

    # assert losses are not same with original ones.
    df_org = pd.read_csv(
        _OUTPUT_DIR / "pretrained_model/log.csv",
        header=0,
        index_col=None,
        skipinitialspace=True,
    )
    assert len(df) == len(df_org)
    pretrained_losses = df.loc[:, "validation_loss"].to_numpy()
    org_losses = df_org.loc[:, "validation_loss"].to_numpy()

    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(pretrained_losses, org_losses)
