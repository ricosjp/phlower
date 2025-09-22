import pathlib
import random
import shutil
from collections.abc import Callable
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch.multiprocessing as mp
import yaml
from phlower.io import PhlowerDirectory, PhlowerYamlFile, select_snapshot_file
from phlower.services.trainer import PhlowerTrainer
from phlower.settings import PhlowerSetting
from phlower.settings._trainer_setting import TrainerInitializerSetting
from phlower.utils.exceptions import (
    PhlowerNaNDetectedError,
    PhlowerRestartTrainingCompletedError,
)
from phlower.utils.typing import IPhlowerHandler

_OUTPUT_DIR = pathlib.Path(__file__).parent / "_out"
_OUTPUT_NAN_DIR = pathlib.Path(__file__).parent / "_out_nan"
_OUTPUT_SLIDING_DIR = pathlib.Path(__file__).parent / "_out_time_series"
_SETTINGS_DIR = pathlib.Path(__file__).parent / "data"


def _initialize_tmp_output_directory(output_dir: pathlib.Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    with (output_dir / ".gitignore").open("w") as fw:
        fw.write("*")


def create_sample_preprocessed_data(output_base_dir: pathlib.Path):
    random.seed(11)
    np.random.seed(11)
    _initialize_tmp_output_directory(output_base_dir)

    base_preprocessed_dir = output_base_dir / "preprocessed"
    base_preprocessed_dir.mkdir()

    n_cases = 4
    dtype = np.float32
    for i in range(n_cases):
        n_nodes = 100 * (i + 1)
        preprocessed_dir = base_preprocessed_dir / f"case_{i}"
        preprocessed_dir.mkdir(exist_ok=True)

        nodal_initial_u = np.random.rand(n_nodes, 3, 1)
        np.save(
            preprocessed_dir / "nodal_initial_u.npy",
            nodal_initial_u.astype(dtype),
        )

        nodal_last_u = np.random.rand(n_nodes, 3, 1)
        np.save(
            preprocessed_dir / "nodal_last_u.npy", nodal_last_u.astype(dtype)
        )

        nodal_initial_p = np.random.rand(n_nodes, 1)
        np.save(
            preprocessed_dir / "nodal_initial_p.npy",
            nodal_initial_p.astype(dtype),
        )
        nodal_last_p = np.random.rand(n_nodes, 1)
        np.save(
            preprocessed_dir / "nodal_last_p.npy", nodal_last_p.astype(dtype)
        )

        rng = np.random.default_rng()
        nodal_nadj = sp.random(n_nodes, n_nodes, density=0.1, random_state=rng)
        sp.save_npz(
            preprocessed_dir / "nodal_nadj", nodal_nadj.tocoo().astype(dtype)
        )

        (preprocessed_dir / "preprocessed").touch()


@pytest.fixture(scope="module")
def prepare_sample_preprocessed_files():
    create_sample_preprocessed_data(_OUTPUT_DIR)


@pytest.fixture(scope="module")
def prepare_sample_preprocessed_files_with_nan_values():
    create_sample_preprocessed_data(_OUTPUT_NAN_DIR)

    u_file = _OUTPUT_NAN_DIR / "preprocessed/case_0/nodal_initial_u.npy"
    assert u_file.exists()
    u = np.load(u_file)
    u[0] = np.nan
    np.save(u_file, u)


@pytest.fixture(scope="module")
def simple_training(prepare_sample_preprocessed_files: None) -> float:
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
    assert loss > 0.0


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
    simple_training: float,
    perform_restart: Callable[[int | None], None],
):
    with pytest.raises(PhlowerRestartTrainingCompletedError):
        perform_restart()


def test__last_epoch_is_update_after_restart(
    simple_training: float,
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


def test__not_recursive_load_restart_setting(
    simple_training: float,
):
    dummy_reference_directory = _OUTPUT_DIR / "dummy_model"
    if dummy_reference_directory.exists():
        shutil.rmtree(dummy_reference_directory)
    shutil.copytree(_OUTPUT_DIR / "model", dummy_reference_directory)

    with open(_SETTINGS_DIR / "train.yml") as fr:
        content = yaml.load(fr, Loader=yaml.SafeLoader)
    content["training"]["n_epoch"] = 15
    content["training"]["initializer_setting"] = {
        "type_name": "restart",
        "reference_directory": "dummy",
    }
    PhlowerYamlFile.save(
        dummy_reference_directory,
        "model",
        content,
        allow_overwrite=True,
    )

    with mock.patch.object(
        PhlowerTrainer,
        "restart_from",
        wraps=PhlowerTrainer.restart_from,
    ) as mocked_reinit:
        _ = PhlowerTrainer.restart_from(dummy_reference_directory)

        assert mocked_reinit.call_count == 1


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


def test__faster_training_with_on_memory_dataset(
    simple_training: float,
    prepare_sample_preprocessed_files: None,
) -> float:
    phlower_path = PhlowerDirectory(_OUTPUT_DIR)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(_SETTINGS_DIR / "train_on_memory.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_DIR / "model_on_memory"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    _ = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )

    # load lazy_load csv
    df_lazy = pd.read_csv(
        _OUTPUT_DIR / "model/log.csv",
        header=0,
        index_col=None,
        skipinitialspace=True,
    )
    lazy_time = df_lazy.loc[:, "elapsed_time"].to_numpy().max()

    df = pd.read_csv(
        output_directory / "log.csv",
        header=0,
        index_col=None,
        skipinitialspace=True,
    )
    on_memory_time = df.loc[:, "elapsed_time"].to_numpy().max()

    # NOTE: the number 1.5 is arbitrary, but it should be larger than 1.0
    assert lazy_time > on_memory_time * 1.5


@pytest.mark.parametrize(
    "name, allow_overwrite, success",
    [
        ("EarlyStopping", True, True),
        ("EarlyStopping", False, False),
        ("DummyHandler", True, True),
        ("DummyHandler", False, True),
    ],
)
def test__attach_hanlder(
    name: str,
    allow_overwrite: bool,
    success: bool,
):
    setting = PhlowerSetting.read_yaml(_SETTINGS_DIR / "train.yml")
    trainer = PhlowerTrainer.from_setting(setting)

    mocked = mock.Mock(spec=IPhlowerHandler)

    if success:
        trainer.attach_handler(name, mocked, allow_overwrite=allow_overwrite)
        assert name in trainer._handlers._handlers
    else:
        with pytest.raises(ValueError) as ex:
            trainer.attach_handler(
                name, mocked, allow_overwrite=allow_overwrite
            )
        assert str(ex.value) == f"Handler named {name} is already attached."


# region Test for dumped details in training log


def simple_training_with_yaml(
    setting: PhlowerSetting, output_directory: pathlib.Path
):
    phlower_path = PhlowerDirectory(_OUTPUT_DIR)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    trainer = PhlowerTrainer.from_setting(setting)
    if output_directory.exists():
        shutil.rmtree(output_directory)

    _ = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )


@pytest.mark.parametrize(
    "yaml_file, output_directory",
    [
        (
            _SETTINGS_DIR / "train_for_multiple_targets.yml",
            _OUTPUT_DIR / "model_for_multiple_targets",
        ),
        (
            _SETTINGS_DIR / "train_for_multiple_targets_with_weights.yml",
            _OUTPUT_DIR / "model_for_multiple_targets_with_weights",
        ),
        (
            _SETTINGS_DIR / "train_for_multiple_targets_not_evaluating.yml",
            _OUTPUT_DIR / "model_for_multiple_targets_not_evaluating",
        ),
    ],
)
def test__dumped_details_training_log(
    yaml_file: pathlib.Path,
    output_directory: pathlib.Path,
    prepare_sample_preprocessed_files: None,
):
    setting = PhlowerSetting.read_yaml(yaml_file)

    simple_training_with_yaml(setting, output_directory=output_directory)

    log_csv = output_directory / "log.csv"
    assert log_csv.exists()
    df = pd.read_csv(
        log_csv,
        header=0,
        index_col=None,
        skipinitialspace=True,
    )

    train_loss = df.loc[:, "train_loss"].to_numpy()
    train_loss_details = df.filter(regex=r"^details_tr")

    # check train loss details
    loss_weights = setting.training.loss_setting.name2weight
    if loss_weights is None:
        weights = dict.fromkeys(train_loss_details.columns, 1.0)
    else:
        weights = {
            name: loss_weights[name.removeprefix("details_tr/")]
            for name in train_loss_details.columns
        }

    _details = np.stack(
        [train_loss_details.loc[:, k].to_numpy() * weights[k] for k in weights],
        axis=1,
    )
    np.testing.assert_array_almost_equal(
        train_loss, np.sum(_details, axis=1), decimal=5
    )

    # check validation loss
    validation_loss = df.loc[:, "validation_loss"].to_numpy()
    validation_loss_details = df.filter(regex=r"^details_val")
    if loss_weights is None:
        weights = dict.fromkeys(validation_loss_details.columns, 1.0)
    else:
        weights = {
            name: loss_weights[name.removeprefix("details_val/")]
            for name in validation_loss_details.columns
        }

    _vl_details = np.stack(
        [
            validation_loss_details.loc[:, k].to_numpy() * weights[k]
            for k in weights
        ],
        axis=1,
    )
    np.testing.assert_array_almost_equal(
        validation_loss, np.sum(_vl_details, axis=1), decimal=5
    )


# endregion


def test__train_with_raise_nan_detected(
    prepare_sample_preprocessed_files_with_nan_values: None,
):
    phlower_path = PhlowerDirectory(_OUTPUT_NAN_DIR)
    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(
        _SETTINGS_DIR / "handlers/raise_nan_detected_error.yml"
    )

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_DIR / "model_w_nan"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    with pytest.raises(PhlowerNaNDetectedError):
        trainer.train(
            train_directories=preprocessed_directories,
            validation_directories=preprocessed_directories,
            output_directory=output_directory,
        )


def test__run_inplace_operation_with_inference_mode(
    prepare_sample_preprocessed_files: None,
):
    # This test checks if inplace operation works without error
    # If torch.inference_mode() is called after DataLoader,
    # this test fails.

    phlower_path = PhlowerDirectory(_OUTPUT_DIR)

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(_SETTINGS_DIR / "train_inplace.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_DIR / "model_w_inplace"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    _ = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )


# region: test for time series sliding window


@pytest.fixture(scope="module")
def prepare_time_sliding_sample_preprocessed_files():
    output_base_dir = _OUTPUT_SLIDING_DIR
    random.seed(11)
    np.random.seed(11)
    _initialize_tmp_output_directory(output_base_dir)

    base_preprocessed_dir = output_base_dir / "preprocessed"
    base_preprocessed_dir.mkdir()

    n_cases = 3
    n_time_series = 10
    dtype = np.float32
    for i in range(n_cases):
        n_nodes = 100 * (i + 1)
        preprocessed_dir = base_preprocessed_dir / f"case_{i}"
        preprocessed_dir.mkdir(exist_ok=True)

        nodal_initial_u = np.random.rand(n_time_series, n_nodes, 3, 1)
        np.save(
            preprocessed_dir / "nodal_time_series_u.npy",
            nodal_initial_u.astype(dtype),
        )

        nodal_u = np.random.rand(n_time_series, n_nodes, 3, 1)
        np.save(preprocessed_dir / "nodal_u.npy", nodal_u.astype(dtype))

        (preprocessed_dir / "preprocessed").touch()


def test__train_with_sliding_window(
    prepare_time_sliding_sample_preprocessed_files: None,
):
    phlower_path = PhlowerDirectory(_OUTPUT_SLIDING_DIR)
    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(
        _SETTINGS_DIR / "train_for_time_series_sliding.yml"
    )

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_SLIDING_DIR / "model_sliding_window"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )


def test__n_call_times_when_time_sliding(
    prepare_time_sliding_sample_preprocessed_files: None,
):
    phlower_path = PhlowerDirectory(_OUTPUT_SLIDING_DIR)
    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(
        _SETTINGS_DIR / "train_for_time_series_sliding.yml"
    )

    # --- check the setting
    n_time_series = 10
    n_offset = 2
    n_size = 3
    n_stride = 1
    # ----

    n_data = len(preprocessed_directories)
    n_length = 1 + (n_time_series - n_offset - n_size) // n_stride
    n_expected_called = setting.training.n_epoch * n_data * n_length

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_SLIDING_DIR / "model_sliding_window"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    with mock.patch(
        "phlower.services.trainer._runners._trainer._training_batch_step_w_slide"
    ) as mocked:
        mocked.side_effect = lambda x, y, z, w: (0.0, {})

        trainer.train(
            train_directories=preprocessed_directories,
            validation_directories=preprocessed_directories,
            output_directory=output_directory,
        )
        assert mocked.call_count == n_expected_called


# endregion


# region: test for parallel training


@pytest.fixture(scope="module")
def simple_distributed_parallel_training(
    prepare_sample_preprocessed_files: None,
) -> pathlib.Path:
    phlower_path = PhlowerDirectory(_OUTPUT_DIR / "preprocessed")

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(_SETTINGS_DIR / "train_ddp_gloo.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_DIR / "model_ddp_gloo"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    world_size = setting.training.parallel_setting.world_size
    mp.spawn(
        trainer.train_ddp,
        args=(
            world_size,
            output_directory,
            preprocessed_directories,
            preprocessed_directories,
        ),
        nprocs=world_size,
        join=True,
    )
    assert (output_directory / "log.csv").exists()
    return output_directory


def test__ddp_training_loss(
    simple_distributed_parallel_training: pathlib.Path,
):
    output_directory = simple_distributed_parallel_training

    df = pd.read_csv(
        output_directory / "log.csv", skipinitialspace=True, header=0
    )
    first_loss = df.loc[0, "train_loss"]
    last_loss = df.loc[df.index[-1], "train_loss"]
    assert first_loss > last_loss


def test__start_pretrained_training_after_dpp_training(
    prepare_sample_preprocessed_files: None,
    simple_distributed_parallel_training: pathlib.Path,
) -> None:
    ddp_output_dir = simple_distributed_parallel_training
    phlower_path = PhlowerDirectory(_OUTPUT_DIR / "preprocessed")

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml(_SETTINGS_DIR / "train_ddp_gloo.yml")
    setting = setting.model_copy(
        update={
            "training": setting.training.model_copy(
                update={
                    "initializer_setting": TrainerInitializerSetting(
                        type_name="pretrained",
                        reference_directory=ddp_output_dir,
                    ),
                }
            )
        }
    )
    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = _OUTPUT_DIR / "pretrained_ddp_model"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    world_size = setting.training.parallel_setting.world_size
    mp.spawn(
        trainer.train_ddp,
        args=(
            world_size,
            output_directory,
            preprocessed_directories,
            preprocessed_directories,
        ),
        nprocs=world_size,
        join=True,
    )

    df_pretrained = pd.read_csv(
        output_directory / "log.csv", skipinitialspace=True, header=0
    )
    df_ddp = pd.read_csv(
        ddp_output_dir / "log.csv", skipinitialspace=True, header=0
    )

    assert (
        df_pretrained.loc[:, "validation_loss"].min()
        < df_ddp.loc[:, "validation_loss"].min()
    )


# endregion
