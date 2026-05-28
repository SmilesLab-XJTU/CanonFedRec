import numpy as np
import scipy.sparse as sp
try:
    from sparsesvd import sparsesvd
    _HAS_SPARSESVD = True
except ModuleNotFoundError:
    sparsesvd = None
    _HAS_SPARSESVD = False

import torch
from torch import nn

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
