from dataclasses import dataclass

import pandas as pd
import torch


@dataclass
class BaseData:
    rating_train: pd.Series
    rating_val: pd.Series
    rating_train_val: pd.Series
    rating_test: pd.Series


@dataclass
class BaseDataInfo:
    num_users: int
    num_items: int
    
    
@dataclass
class GraphModelData(BaseData):
    edge_index: torch.LongTensor
    

    
    

    