from typing import Callable, Union

import numpy as np
import scipy.sparse as sp
import torch

ArrayDataType = Union[np.ndarray, sp.coo_matrix, sp.csr_matrix, sp.csc_matrix]

DenseArrayType = np.ndarray

SparseArrayType = Union[sp.coo_matrix, sp.csr_matrix, sp.csc_matrix]

LossFunctionType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
