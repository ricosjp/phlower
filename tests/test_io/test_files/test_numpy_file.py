import pathlib
import secrets

import numpy as np
import pytest
import scipy.sparse as sp
from phlower.io import PhlowerNumpyFile
from phlower.utils.typing import ArrayDataType

TEST_ENCRYPT_KEY = secrets.token_bytes(32)


@pytest.mark.parametrize(
    "path, ext",
    [
        ("./sample/sample.npy", ".npy"),
        ("./sample/sample.npy.enc", ".npy.enc"),
        ("./sample/sample.npz", ".npz"),
        ("./sample/sample.npz.enc", ".npz.enc"),
    ],
)
def test__check_extension_type(path: str, ext: str):
    path = pathlib.Path(path)
    phlower_path = PhlowerNumpyFile(path)
    assert phlower_path._ext_type.value == ext
    assert phlower_path.file_path == path


@pytest.mark.parametrize(
    "path", [("./sample/sample.pkl"), ("./sample/sample.pkl.enc")]
)
def test__check_error_extension_type(path: str):
    path = pathlib.Path(path)
    with pytest.raises(NotImplementedError):
        _ = PhlowerNumpyFile(path)


@pytest.mark.parametrize(
    "path, enc",
    [
        ("./sample/sample.npy", False),
        ("./sample/sample.npz", False),
        ("./sample/sample.npy.enc", True),
        ("./sample/sample.npz.enc", True),
    ],
)
def test__is_encrypted(path: str, enc: bool):
    path = pathlib.Path(path)
    phlower_path = PhlowerNumpyFile(path)
    assert phlower_path.is_encrypted == enc


@pytest.mark.parametrize(
    "encrypt_key, decrypt_key",
    [
        (None, None),
        (None, TEST_ENCRYPT_KEY),
        (TEST_ENCRYPT_KEY, TEST_ENCRYPT_KEY),
    ],
)
def test__save_npy_and_load(
    encrypt_key: bytes | None,
    decrypt_key: bytes | None,
    setup_test_dir: pathlib.Path,
):
    sample_array = np.random.rand(3, 4)

    saved_path = PhlowerNumpyFile.save(
        output_directory=setup_test_dir,
        file_basename="sample",
        data=sample_array,
        allow_overwrite=True,
        encrypt_key=encrypt_key,
    )

    assert saved_path.file_path.exists()

    loaded_array: np.ndarray = saved_path.load(decrypt_key=decrypt_key)
    np.testing.assert_array_almost_equal(loaded_array.to_numpy(), sample_array)


@pytest.mark.parametrize(
    "encrypt_key, decrypt_key",
    [
        (None, None),
        (None, TEST_ENCRYPT_KEY),
        (TEST_ENCRYPT_KEY, TEST_ENCRYPT_KEY),
    ],
)
def test__save_npz_and_load(
    encrypt_key: bytes | None,
    decrypt_key: bytes | None,
    setup_test_dir: pathlib.Path,
):
    rng = np.random.default_rng()
    sample_array = sp.random(5, 5, density=0.1, random_state=rng)

    saved_path = PhlowerNumpyFile.save(
        output_directory=setup_test_dir,
        file_basename="sample",
        data=sample_array,
        allow_overwrite=True,
        encrypt_key=encrypt_key,
    )

    assert saved_path.file_path.exists()

    loaded_array = saved_path.load(decrypt_key=decrypt_key)
    np.testing.assert_array_almost_equal(
        loaded_array.to_numpy().todense(), sample_array.todense()
    )


@pytest.mark.parametrize(
    "data", [np.random.rand(3, 5), sp.random(5, 5, density=0.1)]
)
def test__cannnot_load_encrypt_data_without_key(
    data: list[ArrayDataType], setup_test_dir: pathlib.Path
):
    saved_path = PhlowerNumpyFile.save(
        output_directory=setup_test_dir,
        file_basename="sample",
        data=data,
        allow_overwrite=True,
        encrypt_key=TEST_ENCRYPT_KEY,
    )

    with pytest.raises(ValueError):
        _ = saved_path.load()


def test__save_not_allowed_overwrite(setup_test_dir: pathlib.Path):
    sample_array = np.random.rand(3, 4)

    path: pathlib.Path = setup_test_dir / "sample.npy"
    path.touch(exist_ok=True)

    with pytest.raises(FileExistsError):
        PhlowerNumpyFile.save(
            output_directory=setup_test_dir,
            data=sample_array,
            file_basename="sample",
            allow_overwrite=False,
        )
