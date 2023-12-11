from dataclasses import dataclass

import pandas as pd
import torch

@dataclass
class LightGCNData:
    edge_index: torch.LongTensor
    train: pd.Series
    valid: pd.Series
    test: pd.Series
    n_users: int
    n_items: int