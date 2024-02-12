import os

import numpy as np
import pandas as pd
import torch

from config import Config
from data.datamodel import BaseData, GraphModelData, MFModelData


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config

    def process(self) -> BaseData:
        rating_df = self._load_raw_data(self.config.dataset)
        rating_df = self._filter_user_item(rating_df, self.config.min_user_cnt, self.config.min_item_cnt)
        rating_df = self._convert_id2index(rating_df)
        num_users = rating_df['user_id'].nunique()
        num_items = rating_df['item_id'].nunique()
        rating_train, rating_val, rating_test = self._split_data(
            rating_df, self.config.split_method, self.config.test_ratio, self.config.val_ratio
        )
        print(f'Number of users: {num_users}')
        print(f'Number of items: {num_items}')
        print(f'Number of ratings: {len(rating_df)}')
        if self.config.model in ['mf']:
            u_i_index = self._get_user_item_index(rating_train)

            return MFModelData(
                num_users=num_users,
                num_items=num_items,
                rating_train=rating_train,
                rating_val=rating_val,
                rating_test=rating_test,
                u_i_index=u_i_index,
            )
        elif self.config.model in ['lightgcn']:
            u_i_index = self._get_user_item_index(rating_train)
            adj_mat = self._get_adj_mat(u_i_index, num_users, num_items)

            return GraphModelData(
                num_users=num_users,
                num_items=num_items,
                rating_train=rating_train,
                rating_val=rating_val,
                rating_test=rating_test,
                u_i_index=u_i_index,
                adj_mat=adj_mat,
            )
        else:
            raise NotImplementedError('Other models are not implemented.')

    def _load_raw_data(self, dataset: str) -> pd.DataFrame:
        data_dir = os.path.join(f'../dataset/{dataset}')
        print(f'Load "{dataset}" data from "{data_dir}"')
        if dataset == 'movielens':
            rating_df = pd.read_csv(
                os.path.join(data_dir, 'ratings.dat'), sep='::', engine='python', header=None, encoding='latin1'
            ).iloc[:, [0, 1, 3]]
            rating_df.columns = ['user_id', 'item_id', 'timestamp']
            rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'], unit='s')
        elif dataset == 'gowalla':
            rating_df = pd.read_csv(os.path.join(data_dir, 'Gowalla_totalCheckins.txt'), sep='\t', header=None).iloc[
                :, [0, 4, 1]
            ]
            rating_df.columns = ['user_id', 'item_id', 'timestamp']
            rating_df['timestamp'] = pd.to_datetime(rating_df['timestamp'], utc=True).dt.tz_localize(None)
        elif dataset == 'yelp':
            rating_df = pd.read_csv(os.path.join(data_dir, 'yelp2018.inter'), sep='\t', engine='python').iloc[
                :, [0, 1, 3]
            ]
            rating_df.columns = ['user_id', 'item_id', 'timestamp']
        else:
            raise NotImplementedError('Other datasets are not available.')
        rating_df = rating_df.sort_values('timestamp')

        return rating_df

    def _filter_user_item(self, rating_df: pd.DataFrame, min_user_cnt: int, min_item_cnt: int) -> pd.DataFrame:
        rating_df = rating_df.drop_duplicates(subset=['user_id', 'item_id'])
        print(f'Filter users < {min_user_cnt} ratings, items < {min_item_cnt} ratings')
        while True:
            user_cnt = rating_df['user_id'].value_counts()
            item_cnt = rating_df['item_id'].value_counts()
            valid_users = user_cnt[user_cnt >= min_user_cnt].index.values
            valid_items = item_cnt[item_cnt >= min_item_cnt].index.values
            filtered_df = rating_df[(rating_df['user_id'].isin(valid_users)) & (rating_df['item_id'].isin(valid_items))]
            if len(filtered_df) == len(rating_df):
                break
            rating_df = filtered_df

        return filtered_df

    def _convert_id2index(self, rating_df: pd.DataFrame) -> pd.DataFrame:
        print('Convert user_id, item_id to index.')
        user2idx = {v: i for i, v in enumerate(rating_df['user_id'].unique())}
        item2idx = {v: i for i, v in enumerate(rating_df['item_id'].unique())}
        rating_df['user_id'] = rating_df['user_id'].map(user2idx)
        rating_df['item_id'] = rating_df['item_id'].map(item2idx)

        return rating_df

    def _split_data(
        self, rating_df: pd.DataFrame, split_method: str, test_ratio: float, val_ratio: float
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
        if split_method == 'random':
            rating_total = rating_df.groupby('user_id')['item_id'].apply(np.random.permutation)
            rating_train = dict()
            rating_val = dict()
            rating_test = dict()
            for user, items in rating_total.items():
                n_test = max(1, int(items.size * test_ratio))
                n_val = max(1, int(items.size * val_ratio))
                rating_test[user] = items[:n_test]
                rating_val[user] = items[n_test : n_test + n_val]
                rating_train[user] = items[n_test + n_val :]
        elif split_method == 'leave_one_out':
            rating_total = rating_df.groupby('user_id')['item_id'].apply(np.array)
            rating_train = dict()
            rating_val = dict()
            rating_test = dict()
            for user, items in rating_total.items():
                rating_test[user] = np.array([items[-1]])
                rating_val[user] = np.array([items[-2]])
                rating_train[user] = items[:-2]
        else:
            raise NotImplementedError('Other splits are not implmented.')

        return rating_train, rating_val, rating_test

    def _get_user_item_index(self, rating: dict[int, np.ndarray]) -> np.ndarray:
        return pd.Series(rating).explode().reset_index().values.astype(np.int64)

    def _get_adj_mat(self, u_i_index: np.ndarray, num_users: int, num_items: int) -> torch.sparse.Tensor:
        u_i_index = u_i_index.copy()
        u_i_index[:, 1] += num_users
        u_index, i_index = u_i_index[:, 0], u_i_index[:, 1]
        i_u_index = np.stack([i_index, u_index], axis=1)
        indices = torch.tensor(np.concatenate([u_i_index, i_u_index], axis=0).T, dtype=torch.int64)
        values = torch.ones(indices.shape[1])
        size = (num_users + num_items, num_users + num_items)

        return torch.sparse_coo_tensor(indices, values, size).coalesce()
