from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import torch

ArrayDataType = np.ndarray | sp.coo_matrix | sp.csr_matrix | sp.csc_matrix

DenseArrayType = np.ndarray

SparseArrayType = sp.coo_matrix | sp.csr_matrix | sp.csc_matrix

LossFunctionType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
