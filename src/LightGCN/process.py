import os
import pickle
import pandas as pd
import numpy as np
import torch
from numpy.typing import NDArray

class DataProcessor:
    def __init__(self,
                 dataset: str,
                 min_user_cnt: int,
                 min_item_cnt: int,
                 test_ratio: float,
                 valid_ratio: float):
        self.dataset = dataset
        self.data_dir = os.path.join('dataset/', self.dataset)
        self.min_user_counts = min_user_cnt
        self.min_item_counts = min_item_cnt
        self.test_ratio = test_ratio
        self.valid_ratio = valid_ratio
        
    def process(self) -> tuple[torch.LongTensor, pd.Series, pd.Series, pd.Series, int, int]:
        df = self.load_df()
        df, n_users, n_items = self.process_df(df)
        total = df.groupby('user_id')['item_id'].apply(np.array)
        train_val, test = self.train_test_split(total, self.test_ratio)
        train, val = self.train_test_split(train_val, self.valid_ratio)
        edge_index = self.convert_to_edge_index(train, n_users)
        
        return edge_index, train, val, test, n_users, n_items
    
    def load_df(self) -> pd.DataFrame:
        if self.dataset == 'movielens':
            df = pd.read_csv(os.path.join(self.data_dir, 'ratings.dat'),
                            sep='::',
                            engine='python',
                            header=None,
                            encoding='latin1').iloc[:, :2]
            df.columns = ['user_id', 'item_id']
            return df
        else:
            raise NotImplementedError('Other datasets are not available.')

    def process_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        df = df.drop_duplicates(subset=['user_id', 'item_id'])
        
        user_cnt = df['user_id'].value_counts()
        valid_users = user_cnt[user_cnt >= self.min_user_counts].index.values
        item_cnt = df['item_id'].value_counts()
        valid_items = item_cnt[item_cnt >= self.min_item_counts].index.values
        df = df[(df['user_id'].isin(valid_users)) & (df['item_id'].isin(valid_items))]

        user2idx = {v: i for i, v in enumerate(df['user_id'].unique())}
        item2idx = {v: i for i, v in enumerate(df['item_id'].unique())}
        mapper = {'user2idx': user2idx, 'item2idx': item2idx}
        df['user_id'] = df['user_id'].map(user2idx)
        df['item_id'] = df['item_id'].map(item2idx)

        df.to_csv(os.path.join(self.data_dir, 'processed_df.csv'), index=False)
        with open(os.path.join(self.data_dir, 'mapper.pkl'), 'wb') as f:
            pickle.dump(mapper, f)

        n_users = len(user2idx)
        n_items = len(item2idx)

        return df, n_users, n_items
    
    def train_test_split(self, total: pd.Series, ratio: float) -> tuple[pd.Series, pd.Series]:
        total = total.apply(np.random.permutation)
        train = list()
        test = list()
        for user in total.index:
            items = total[user]
            n_test = max(1, int(items.size * ratio))
            test.append(items[:n_test])
            train.append(items[n_test:])
        train = pd.Series(train)
        test = pd.Series(test)
        train.index = total.index
        test.index = total.index
        
        return train, test
    
    def convert_to_edge_index(self, train: pd.Series, n_users: int) -> torch.LongTensor:
        train_explode = train.explode()
        user_index = train_explode.index.values.astype(np.int64)
        item_index = train_explode.values.astype(np.int64) + n_users
        edge_index = torch.from_numpy(np.stack([user_index, item_index], axis=0))
        edge_index_rev = torch.from_numpy(np.stack([item_index, user_index], axis=0))
    
        return torch.cat([edge_index, edge_index_rev], dim=1)


