import io
import pathlib

import numpy as np
import scipy.sparse as sp

from phlower import utils
from phlower._base.array import IPhlowerArray, phlower_array
from phlower.utils.enums import PhlowerFileExtType
from phlower.utils.typing import ArrayDataType

from .interface import IPhlowerNumpyFile

# NOTE
# IF branches due to difference of extensions (.npy, .npy.enc,..) increases,
# introduce new interface class to absorb it.


class PhlowerNumpyFile(IPhlowerNumpyFile):
    def __init__(self, path: pathlib.Path) -> None:
        ext = self._check_extension_type(path)
        self._path = path
        self._ext_type = ext

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self._path}"

    def _check_extension_type(self, path: pathlib.Path) -> PhlowerFileExtType:
        extensions = [
            PhlowerFileExtType.NPY,
            PhlowerFileExtType.NPYENC,
            PhlowerFileExtType.NPZ,
            PhlowerFileExtType.NPZENC,
        ]
        for ext in extensions:
            if path.name.endswith(ext.value):
                return ext

        raise NotImplementedError(f"Unknown file extension: {path}")

    @property
    def is_encrypted(self) -> bool:
        if self._ext_type == PhlowerFileExtType.NPYENC:
            return True
        if self._ext_type == PhlowerFileExtType.NPZENC:
            return True

        return False

    @property
    def file_extension(self):
        return self._ext_type.value

    @property
    def file_path(self) -> pathlib.Path:
        return self._path

    def load(
        self, *, check_nan: bool = False, decrypt_key: bytes = None
    ) -> IPhlowerArray:
        loaded_data = self._load(decrypt_key=decrypt_key)
        if check_nan and np.any(np.isnan(loaded_data)):
            raise ValueError(f"NaN found in {self._path}")

        return phlower_array(loaded_data)

    def _load(self, *, decrypt_key: bytes = None) -> ArrayDataType:
        if self._ext_type == PhlowerFileExtType.NPY:
            return self._load_npy()

        if self._ext_type == PhlowerFileExtType.NPYENC:
            return self._load_npy_enc(decrypt_key)

        if self._ext_type == PhlowerFileExtType.NPZ:
            return self._load_npz()

        if self._ext_type == PhlowerFileExtType.NPZENC:
            return self._load_npz_enc(decrypt_key)

        raise NotImplementedError(
            "Loading function for this file extenstion is not implemented: "
            f"{self._path}"
        )

    def _load_npy(self):
        return np.load(self._path)

    def _load_npz(self):
        return sp.load_npz(self._path)

    def _load_npy_enc(self, decrypt_key: bytes):
        if decrypt_key is None:
            raise ValueError("Key is None. Cannot decrypt encrypted file.")

        return np.load(utils.decrypt_file(decrypt_key, self._path))

    def _load_npz_enc(self, decrypt_key: bytes):
        if decrypt_key is None:
            raise ValueError("Key is None. Cannot decrypt encrypted file.")

        return sp.load_npz(utils.decrypt_file(decrypt_key, self._path))

    def save(
        self,
        data: ArrayDataType,
        *,
        encrypt_key: bytes = None,
        overwrite: bool = True,
    ) -> None:
        if not overwrite:
            if self.file_path.exists():
                raise FileExistsError(f"{self._path} already exists")

        if self.is_encrypted:
            if encrypt_key is None:
                raise ValueError(
                    f"key is empty when encrpting file: {self._path}"
                )

        file_basename = self._path.name.removesuffix(self._ext_type.value)
        _save_variable(
            self._path.parent,
            file_basename=file_basename,
            data=data,
            encrypt_key=encrypt_key,
        )


# HACK NEED TO REAFCTOR !!!!!!


def _save_variable(
    output_directory: pathlib.Path,
    file_basename: str,
    data: np.ndarray | sp.coo_matrix,
    *,
    dtype: type = np.float32,
    encrypt_key: bytes = None,
):
    """Save variable data.

    Parameters
    ----------
    output_directory: pathlib.Path
        Save directory path.
    file_basename: str
        Save file base name without extenstion.
    data: np.ndarray or scipy.sparse.coo_matrix
        Data to be saved.
    dtype: type, optional
        Data type to be saved.
    encrypt_key: bytes, optional
        Data for encryption.

    Returns
    --------
        None
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    if isinstance(data, np.ndarray):
        if encrypt_key is None:
            save_file_path = output_directory / (file_basename + ".npy")
            np.save(save_file_path, data.astype(dtype))
        else:
            save_file_path = output_directory / (file_basename + ".npy.enc")
            bytesio = io.BytesIO()
            np.save(bytesio, data.astype(dtype))
            utils.encrypt_file(encrypt_key, save_file_path, bytesio)

    elif isinstance(data, (sp.coo_matrix, sp.csr_matrix, sp.csc_matrix)):
        if encrypt_key is None:
            save_file_path = output_directory / (file_basename + ".npz")
            sp.save_npz(save_file_path, data.tocoo().astype(dtype))
        else:
            save_file_path = output_directory / (file_basename + ".npz.enc")
            bytesio = io.BytesIO()
            sp.save_npz(bytesio, data.tocoo().astype(dtype))
            utils.encrypt_file(encrypt_key, save_file_path, bytesio)
    else:
        raise ValueError(f"{file_basename} has unknown type: {data.__class__}")

    print(f"{file_basename} is saved in: {save_file_path}")
    return
