import io
import pathlib

from Cryptodome.Cipher import AES


def encrypt_file(key: bytes, file_path: pathlib.Path, binary: io.BytesIO):
    """Encrypt data and then save to a file.

    Parameters
    ----------
    key: bytes
        Key for encription.
    file_path: str or pathlib.Path
        File path to save.
    binary: io.BytesIO
        Data content.
    """
    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(binary.getvalue())
    with open(file_path, "wb") as f:
        [f.write(x) for x in (cipher.nonce, tag, ciphertext)]


def decrypt_file(
    key: bytes, file_path: pathlib.Path, return_stringio: bool = False
) -> str | bytes:
    """Decrypt data file.

    Parameters
    ----------
    key: bytes
        Key for decryption.
    file_path: str or pathlib.Path
        File path of the encrypted data.
    return_stringio: bool, optional
        If True, return io.StrintIO instead of io.BytesIO.

    Returns
    -------
    str | bytes
        _description_
    """

    with open(file_path, "rb") as f:
        nonce, tag, ciphertext = (f.read(x) for x in (16, 16, -1))
    cipher = AES.new(key, AES.MODE_EAX, nonce)
    if return_stringio:
        return cipher.decrypt_and_verify(ciphertext, tag).decode("utf-8")
    else:
        return io.BytesIO(cipher.decrypt_and_verify(ciphertext, tag))
