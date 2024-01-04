from dataclasses import dataclass

import pandas as pd
import torch


@dataclass
class BaseData:
    num_users: int
    num_items: int
    rating_train: pd.Series
    rating_val: pd.Series
    rating_train_val: pd.Series
    rating_test: pd.Series

    
@dataclass
class GraphModelData(BaseData):
    edge_index: torch.LongTensor
    

    
    

    