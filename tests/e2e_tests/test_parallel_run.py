import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from phlower.io import PhlowerDirectory
from phlower.services.trainer import PhlowerTrainer
from phlower.settings import PhlowerSetting
from phlower.settings._trainer_setting import (
    ParallelSetting,
    TrainerInitializerSetting,
)


@pytest.mark.need_multigpu
@pytest.mark.e2e_test
def test__check_multi_gpu_testing_is_available() -> None:
    assert torch.cuda.device_count() >= 2
    assert torch.cuda.is_available()
    assert torch.backends.cudnn.is_available()
    assert dist.is_nccl_available()


@pytest.fixture(scope="module")
def simple_distributed_parallel_training(
    prepare_sample_preprocessed_files_fixture: pathlib.Path,
) -> pathlib.Path:
    output_dir = prepare_sample_preprocessed_files_fixture
    phlower_path = PhlowerDirectory(output_dir / "preprocessed")

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/train_ddp.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = output_dir / "model_ddp"
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


@pytest.fixture(scope="module")
def simple_same_size_batch_training(
    prepare_sample_preprocessed_files_fixture: pathlib.Path,
) -> pathlib.Path:
    output_dir = prepare_sample_preprocessed_files_fixture
    phlower_path = PhlowerDirectory(output_dir / "preprocessed")

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/train_ddp.yml")
    batch_size = (
        setting.training.batch_size
        * setting.training.parallel_setting.world_size
    )
    setting = setting.model_copy(
        update={
            "training": setting.training.model_copy(
                update={
                    "batch_size": batch_size,
                    "parallel_setting": ParallelSetting(is_active=False),
                }
            )
        }
    )
    assert setting.training.batch_size == batch_size
    assert not setting.training.parallel_setting.is_active

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = output_dir / f"model_batch_{batch_size}"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    loss = trainer.train(
        train_directories=preprocessed_directories,
        validation_directories=preprocessed_directories,
        output_directory=output_directory,
    )
    assert loss > 0
    return output_directory


@pytest.mark.need_multigpu
@pytest.mark.e2e_test
def test_dpp_training_is_faster_than_single_gpu(
    simple_training: pathlib.Path,
    simple_distributed_parallel_training: pathlib.Path,
) -> None:
    single_df = pd.read_csv(
        simple_training / "log.csv", skipinitialspace=True, header=0
    )
    ddp_df = pd.read_csv(
        simple_distributed_parallel_training / "log.csv",
        skipinitialspace=True,
        header=0,
    )

    elapsed_single = single_df.loc[:, "elapsed_time"].to_numpy()[-1]
    elapsed_ddp = ddp_df.loc[:, "elapsed_time"].to_numpy()[-1]

    # at least 2 times faster
    assert elapsed_ddp < elapsed_single


@pytest.mark.need_multigpu
@pytest.mark.e2e_test
def test__start_pretrained_training_after_dpp_training(
    prepare_sample_preprocessed_files_fixture: pathlib.Path,
    simple_distributed_parallel_training: pathlib.Path,
) -> None:
    ddp_output_dir = simple_distributed_parallel_training
    output_dir = prepare_sample_preprocessed_files_fixture
    phlower_path = PhlowerDirectory(output_dir / "preprocessed")

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/train_ddp.yml")
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
    output_directory = output_dir / "pretrained_ddp_model"
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


@pytest.mark.need_multigpu
@pytest.mark.e2e_test
def test__dpp_training_same_as_single_batch_training(
    simple_same_size_batch_training: pathlib.Path,
    simple_distributed_parallel_training: pathlib.Path,
) -> None:
    ddp_df = pd.read_csv(
        simple_distributed_parallel_training / "log.csv",
        skipinitialspace=True,
        header=0,
    )
    batch_df = pd.read_csv(
        simple_same_size_batch_training / "log.csv",
        skipinitialspace=True,
        header=0,
    )

    loss_ddp = ddp_df.loc[:, "validation_loss"].to_numpy()
    loss_batch = batch_df.loc[:, "validation_loss"].to_numpy()

    np.testing.assert_array_almost_equal(loss_ddp, loss_batch, decimal=5)


# region FSDP (Fully Shard Distributed Parallel) tests


@pytest.fixture(scope="module")
def simple_distributed_fsdp_parallel_training(
    prepare_sample_preprocessed_files_fixture: pathlib.Path,
) -> pathlib.Path:
    output_dir = prepare_sample_preprocessed_files_fixture
    phlower_path = PhlowerDirectory(output_dir / "preprocessed")

    preprocessed_directories = list(
        phlower_path.find_directory(
            required_filename="preprocessed", recursive=True
        )
    )

    setting = PhlowerSetting.read_yaml("tests/e2e_tests/data/train_fsdp.yml")

    trainer = PhlowerTrainer.from_setting(setting)
    output_directory = output_dir / "model_fsdp"
    if output_directory.exists():
        shutil.rmtree(output_directory)

    world_size = setting.training.parallel_setting.world_size
    mp.spawn(
        trainer.train_fsdp,
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


@pytest.mark.skip(reason="Share module is not available in FSDP.")
@pytest.mark.need_multigpu
@pytest.mark.e2e_test
def test__fsdp_training_is_enable(
    # simple_training: pathlib.Path,
    simple_distributed_fsdp_parallel_training: pathlib.Path,
) -> None:
    # single_df = pd.read_csv(
    #     simple_training / "log.csv", skipinitialspace=True, header=0
    # )
    ddp_df = pd.read_csv(
        simple_distributed_fsdp_parallel_training / "log.csv",
        skipinitialspace=True,
        header=0,
    )

    assert not ddp_df.empty
    # elapsed_single = single_df.loc[:, "elapsed_time"].to_numpy()[-1]
    # elapsed_ddp = ddp_df.loc[:, "elapsed_time"].to_numpy()[-1]

    # # at least 2 times faster
    # assert elapsed_ddp < elapsed_single


# endregion
