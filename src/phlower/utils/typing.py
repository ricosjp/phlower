from typing import Union, Callable

import torch
import numpy as np
import scipy.sparse as sp

ArrayDataType = Union[np.ndarray, sp.coo_matrix, sp.csr_matrix, sp.csc_matrix]

DenseArrayType = np.ndarray

SparseArrayType = Union[sp.coo_matrix, sp.csr_matrix, sp.csc_matrix]

LossFunctionType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
