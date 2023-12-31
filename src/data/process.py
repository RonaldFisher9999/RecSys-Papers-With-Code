import os

import numpy as np
import pandas as pd
import torch

from config import Config
from data.datamodel import GraphModelData, BaseData


def load_raw_data(dataset: str) -> pd.DataFrame:
    data_dir = os.path.join(f'../dataset/{dataset}')
    print(f'Load "{dataset}" data from "{data_dir}"')
    if dataset == 'movielens':
        rating_df = pd.read_csv(os.path.join(data_dir, 'ratings.dat'),
                        sep='::',
                        engine='python',
                        header=None,
                        encoding='latin1').iloc[:, [0, 1, 3]]
        rating_df.columns = ['user_id', 'item_id', 'timestamp']
        rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'], unit='s')
    elif dataset == 'gowalla':
        rating_df = pd.read_csv(os.path.join(data_dir, 'Gowalla_totalCheckins.txt'),
                                sep='\t', header=None).iloc[:, [0, 4, 1]]
        rating_df.columns = ['user_id', 'item_id', 'timestamp']
        rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'], utc=True).dt.tz_localize(None)
    elif dataset == 'yelp':
        rating_df = pd.read_csv(os.path.join(data_dir, 'yelp2018.inter'), sep='\t', engine='python').iloc[:, [0, 1, 3]]
        rating_df.columns = ['user_id', 'item_id', 'timestamp']
    else:
        raise NotImplementedError('Other datasets are not available.')
    
    return rating_df

def filter_user_item(rating_df: pd.DataFrame,
                     min_user_cnt: int,
                     min_item_cnt: int) -> pd.DataFrame:
    rating_df = rating_df.drop_duplicates(subset=['user_id', 'item_id'])
    print(f'Filter users < {min_user_cnt} ratings, items < {min_item_cnt} ratings')
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
        
def apply_index_mapper(rating_df: pd.DataFrame) -> pd.DataFrame:
    print('Map user_id, item_id to index.')
    user2idx = {v: i for i, v in enumerate(rating_df['user_id'].unique())}
    item2idx = {v: i for i, v in enumerate(rating_df['item_id'].unique())}
    rating_df['user_id'] = rating_df['user_id'].map(user2idx)
    rating_df['item_id'] = rating_df['item_id'].map(item2idx)

    return rating_df
    
def train_test_split(rating_total: pd.Series, ratio: float) -> tuple[pd.Series, pd.Series | None]:
    if ratio == 0:
        return rating_total, None
    else:
        print(f'Split train/test dataset. Split ratio: {ratio}')
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

def get_edge_index(rating: pd.Series, n_users: int) -> torch.LongTensor:
    print('Convert rating data to edge index.')
    rating_explode = rating.explode()
    user_index = rating_explode.index.values.astype(np.int64)
    item_index = rating_explode.values.astype(np.int64) + n_users
    edge_index = torch.from_numpy(np.stack([user_index, item_index], axis=0))
    edge_index_rev = torch.from_numpy(np.stack([item_index, user_index], axis=0))

    return torch.cat([edge_index, edge_index_rev], dim=1)

def process_data(config: Config) -> BaseData:
    print('Process data.')
    rating_df = load_raw_data(config.dataset)
    rating_df = filter_user_item(rating_df, config.min_user_cnt, config.min_item_cnt)
    rating_df = apply_index_mapper(rating_df)
    num_users = rating_df['user_id'].nunique()
    num_items = rating_df['item_id'].nunique()
    rating_total = rating_df.groupby('user_id')['item_id'].apply(np.array)
    rating_train_val, rating_test = train_test_split(rating_total, config.test_ratio)
    rating_train, rating_val = train_test_split(rating_train_val, config.val_ratio)
    print(f'Number of users: {num_users}')
    print(f'Number of items: {num_items}')
    print(f'Number of ratings: {len(rating_df)}')

    if config.model in ['lightgcn']:
        edge_index = get_edge_index(rating_train, num_users)
        data = GraphModelData(
            num_users=num_users,
            num_items=num_items,
            rating_train=rating_train,
            rating_val=rating_val,
            rating_train_val=rating_train_val,
            rating_test=rating_test,
            edge_index=edge_index,
        )
    elif config.model in ['bpr']:
        edge_index = get_edge_index(rating_train, num_users)
        data = GraphModelData(
            num_users=num_users,
            num_items=num_items,
            rating_train=rating_train,
            rating_val=rating_val,
            rating_train_val=rating_train_val,
            rating_test=rating_test,
            edge_index=edge_index,
        )
    else:
        NotImplementedError('Other datasets are not available.')
        
    return data
        
    