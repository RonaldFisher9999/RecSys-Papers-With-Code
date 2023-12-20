from dataclasses import dataclass

import pandas as pd
import torch

@dataclass
class BaseData:
    rating_train: pd.Series
    rating_val: pd.Series | None
    rating_test: pd.Series | None
    

class LightGCNData(BaseData):
    edge_index: torch.LongTensor