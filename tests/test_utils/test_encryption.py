import io
import pathlib
import secrets
import shutil

import pytest

from phlower.utils import decrypt_file, encrypt_file

_OUTPUT_DIR = pathlib.Path(__file__).parent / "_tmp/_encryption"
TEST_ENCRYPT_KEY = secrets.token_bytes(32)


@pytest.fixture(scope="module")
def prepare_empty_direcotry():
    if _OUTPUT_DIR.exists():
        shutil.rmtree(_OUTPUT_DIR)

    _OUTPUT_DIR.mkdir(parents=True)


def test__encrypt_file(prepare_empty_direcotry: None):
    file_path = _OUTPUT_DIR / "sample.txt"
    encrypt_file(
        TEST_ENCRYPT_KEY,
        file_path,
        binary=io.BytesIO("sample_content".encode("utf-8")),  # NOQA
    )

    with pytest.raises(UnicodeDecodeError):
        with open(file_path) as fr:
            _ = fr.read()


def test__decrypt_file(prepare_empty_direcotry: None):
    file_path = _OUTPUT_DIR / "sample.txt"
    content = "sample_content"
    encrypt_file(
        TEST_ENCRYPT_KEY,
        file_path,
        binary=io.BytesIO(content.encode("utf-8")),  # NOQA
    )

    loaded = decrypt_file(TEST_ENCRYPT_KEY, file_path, return_stringio=True)
    assert loaded == content
