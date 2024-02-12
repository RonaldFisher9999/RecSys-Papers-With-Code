from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BaseData:
    num_users: int
    num_items: int
    rating_train: dict[int, np.ndarray]
    rating_val: dict[int, np.ndarray]
    rating_test: dict[int, np.ndarray]


@dataclass
class MFModelData(BaseData):
    u_i_index: np.ndarray


@dataclass
class GraphModelData(BaseData):
    u_i_index: np.ndarray
    adj_mat: torch.sparse.Tensor
