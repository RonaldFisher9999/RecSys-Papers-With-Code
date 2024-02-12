import numpy as np
import pandas as pd
import torch


def get_neg_items(rating: pd.Series, num_users: int, num_items: int, num_neg_samples: int) -> torch.LongTensor:
    print(f'Sample {num_neg_samples} negative items for each positive item.')
    num_total_pos_items = sum(map(np.size, rating.values))
    total_neg_items = np.zeros((num_neg_samples, num_total_pos_items))
    total_items = np.arange(num_items)
    offset = 0
    for user in rating.index:
        pos_items = rating[user]
        candi_items = np.setdiff1d(total_items, pos_items)
        neg_items = candi_items[np.random.randint(0, candi_items.size, pos_items.size * num_neg_samples)]
        total_neg_items[:, offset : offset + pos_items.size] = neg_items.reshape(-1, pos_items.size)
        offset += pos_items.size
    total_neg_items += num_users

    return torch.tensor(total_neg_items, dtype=torch.int64)
