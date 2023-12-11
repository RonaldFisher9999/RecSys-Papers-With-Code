import os

import numpy as np
import pandas as pd
import torch

from data.datamodel import LightGCNData


def load_rating_data(data_dir: str, dataset: str) -> pd.DataFrame:
    if dataset == 'movielens':
        rating_df = pd.read_csv(os.path.join(data_dir, 'ratings.dat'),
                        sep='::',
                        engine='python',
                        header=None,
                        encoding='latin1').iloc[:, :2]
        rating_df.columns = ['user_id', 'item_id']
    elif dataset == 'gowalla':
        rating_df = pd.read_csv(os.path.join(data_dir, 'Gowalla_totalCheckins.txt'),
                                sep='\t', header=None).iloc[:, [0, 4]]
        rating_df.columns = ['user_id', 'item_id']
    else:
        raise NotImplementedError('Other datasets are not available.')
    
    return rating_df

def filter_users_items(rating_df: pd.DataFrame,
                       min_user_cnt: int,
                       min_item_cnt: int) -> pd.DataFrame:
    while True:
        user_cnt = rating_df['user_id'].value_counts()
        item_cnt = rating_df['item_id'].value_counts()
        valid_users = user_cnt[user_cnt >= min_user_cnt].index.values
        valid_items = item_cnt[item_cnt >= min_item_cnt].index.values
        filtered_df = rating_df[(rating_df['user_id'].isin(valid_users))
                              & (rating_df['item_id'].isin(valid_items))]
        if len(filtered_df) == len(rating_df):
            break
        rating_df = filtered_df
    
    return filtered_df
        
    
def process_rating_df(rating_df: pd.DataFrame,
                      min_user_cnt: int,
                      min_item_cnt: int) -> tuple[pd.Series, int, int]:
    rating_df = rating_df.drop_duplicates(subset=['user_id', 'item_id'])
    
    rating_df = filter_users_items(rating_df, min_user_cnt, min_item_cnt)

    user2idx = {v: i for i, v in enumerate(rating_df['user_id'].unique())}
    item2idx = {v: i for i, v in enumerate(rating_df['item_id'].unique())}
    rating_df['user_id'] = rating_df['user_id'].map(user2idx)
    rating_df['item_id'] = rating_df['item_id'].map(item2idx)

    rating_total = rating_df.groupby('user_id')['item_id'].apply(np.array)
    n_users = len(user2idx)
    n_items = len(item2idx)

    return rating_total, n_users, n_items

def train_test_split(rating_total: pd.Series, ratio: float) -> tuple[pd.Series, pd.Series]:
    rating_total = rating_total.apply(np.random.permutation)
    rating_train = dict()
    rating_test = dict()
    for user in rating_total.index:
        items = rating_total[user]
        n_test = max(1, int(items.size * ratio))
        rating_test[user] = items[:n_test]
        rating_train[user] = items[n_test:]
    rating_train = pd.Series(rating_train)
    rating_test = pd.Series(rating_test)
    
    return rating_train, rating_test

def convert_to_edge_index(rating: pd.Series, n_users: int) -> torch.LongTensor:
    rating_explode = rating.explode()
    user_index = rating_explode.index.values.astype(np.int64)
    item_index = rating_explode.values.astype(np.int64) + n_users
    edge_index = torch.from_numpy(np.stack([user_index, item_index], axis=0))
    edge_index_rev = torch.from_numpy(np.stack([item_index, user_index], axis=0))

    return torch.cat([edge_index, edge_index_rev], dim=1)


class LightGCNDataProcessor:
    def __init__(self,
                 dataset: str,
                 min_user_cnt: int,
                 min_item_cnt: int,
                 test_ratio: float,
                 val_ratio: float):
        self.dataset = dataset
        self.data_dir = os.path.join('../dataset/', self.dataset)
        self.min_user_cnt = min_user_cnt
        self.min_item_cnt = min_item_cnt
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        
    def process(self):
        rating_df = load_rating_data(self.data_dir, self.dataset)
        rating_total, n_users, n_items = process_rating_df(rating_df, self.min_user_cnt, self.min_item_cnt)
        rating_tr_val, rating_test = train_test_split(rating_total, self.test_ratio)
        rating_train, rating_val = train_test_split(rating_tr_val, self.val_ratio)
        edge_index = convert_to_edge_index(rating_train, n_users)
        
        return edge_index, rating_train, rating_val, rating_test, n_users, n_items
    