import pathlib
import secrets

import numpy as np
import pytest
import torch

from phlower.io import PhlowerCheckpointFile

TEST_ENCRYPT_KEY = secrets.token_bytes(32)


@pytest.mark.parametrize(
    "path, ext",
    [("./sample/sample.pth", ".pth"), ("./sample/sample.pth.enc", ".pth.enc")],
)
def test__check_extension_type(path: str, ext: str):
    path = pathlib.Path(path)
    siml_path = PhlowerCheckpointFile(path)
    assert siml_path._ext_type.value == ext
    assert siml_path.file_path == path


@pytest.mark.parametrize(
    "path", [("./sample/sample.npy"), ("./sample/sample.npy.enc")]
)
def test__check_error_extension_type(path: str):
    path = pathlib.Path(path)
    with pytest.raises(NotImplementedError):
        _ = PhlowerCheckpointFile(path)


@pytest.mark.parametrize(
    "path, enc",
    [("./sample/sample.pth", False), ("./sample/sample.pth.enc", True)],
)
def test__is_encrypted(path: str, enc: bool):
    path = pathlib.Path(path)
    siml_path = PhlowerCheckpointFile(path)
    assert siml_path.is_encrypted == enc


def test__save_and_load(setup_test_dir: pathlib.Path):
    sample_tensor = torch.tensor(np.random.rand(3, 4))

    saved_path = PhlowerCheckpointFile.save(
        output_directory=setup_test_dir,
        epoch_number=10,
        file_basename="sample",
        data=sample_tensor,
        overwrite=True,
    )

    assert saved_path.file_path.exists()

    loaded_data = saved_path.load(map_location="cpu")
    np.testing.assert_array_almost_equal(loaded_data, sample_tensor)


def test__save_encrypted_and_load(setup_test_dir: pathlib.Path):
    sample_tensor = torch.tensor(np.random.rand(3, 4))

    output_directory = setup_test_dir
    saved_path = PhlowerCheckpointFile.save(
        output_directory=output_directory,
        file_basename="snapshot_epoch_10",
        data=sample_tensor,
        overwrite=True,
        encrypt_key=TEST_ENCRYPT_KEY,
    )

    assert saved_path.file_path.exists()

    loaded_data = saved_path.load(
        map_location="cpu", decrypt_key=TEST_ENCRYPT_KEY
    )
    np.testing.assert_array_almost_equal(loaded_data, sample_tensor)


def test__cannnot_load_without_key(setup_test_dir: pathlib.Path):
    sample_tensor = torch.tensor(np.random.rand(3, 4))

    output_directory = setup_test_dir
    saved_path = PhlowerCheckpointFile.save(
        output_directory=output_directory,
        file_basename="snapshot_epoch_1",
        data=sample_tensor,
        overwrite=True,
        encrypt_key=TEST_ENCRYPT_KEY,
    )

    assert saved_path.file_path.exists()

    with pytest.raises(ValueError, match="Feed key to load encrypted model"):
        _ = saved_path.load(map_location="cpu")


def test__save_not_allowed_overwrite(setup_test_dir: pathlib.Path):
    sample_array = np.random.rand(3, 4)
    sample_tensor = torch.tensor(sample_array)

    path = setup_test_dir / "snapshot_epoch_10.pth"
    path.touch(exist_ok=True)

    with pytest.raises(FileExistsError):
        PhlowerCheckpointFile.save(
            output_directory=setup_test_dir,
            file_basename="snapshot_epoch_10",
            data=sample_tensor,
            overwrite=False,
        )


@pytest.mark.parametrize(
    "path, num_epoch",
    [
        ("./aaa/snapshot_epoch_1.pth", 1),
        ("./aaa/snapshot_epoch_123.pth.enc", 123),
        ("./aaa/snapshot_epoch_30.pth.enc", 30),
    ],
)
def test__get_epoch(path: str, num_epoch: int):
    phlower_path = PhlowerCheckpointFile(path)
    assert phlower_path.epoch == num_epoch


@pytest.mark.parametrize(
    "path",
    [("./aaa/deployed_1.pth"), ("./aaa/model.pth"), ("./aaa/epoch_2.pth")],
)
def test__get_epoch_not_handled(path: str):
    phlower_path = PhlowerCheckpointFile(path)

    with pytest.raises(ValueError):
        _ = phlower_path.epoch
