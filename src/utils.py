from argparse import Namespace
import torch
import numpy as np
import random
import pandas as pd


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_neg_items(rating: pd.Series, n_items: int, num_neg_samples: int) -> torch.LongTensor:
    print(f'{num_neg_samples} negative samples for each positive item')
    num_total_pos_items = sum(map(np.size, rating.values))
    total_neg_items = np.zeros((num_neg_samples, num_total_pos_items))
    total_items = np.arange(n_items)
    offset = 0
    for user in rating.index:
        pos_items = rating[user]
        candi_items = np.setdiff1d(total_items, pos_items)
        neg_items = candi_items[np.random.randint(0, candi_items.size, pos_items.size * num_neg_samples)]
        total_neg_items[:, offset : offset + pos_items.size] = neg_items.reshape(-1,  pos_items.size)
        offset += pos_items.size
    
    return torch.tensor(total_neg_items, dtype=torch.int64)

def recall_k(y_true: list[list[int]], y_pred: list[list[int]], k: int) -> float:
    total_recall = 0
    for true, pred in zip(y_true, y_pred):
        recall = len(set(true) & set(pred[:k])) / min(len(true), k)
        total_recall += recall
    
    return round(total_recall / len(y_true), 6)

def ndcg_k(y_true: list[list[int]], y_pred: list[list[int]],  k: int):
    total_ndcg = 0
    for true, pred in zip(y_true, y_pred):
        true = set(true)
        hit_list = np.array([1 if x in true else 0 for x in pred[:k]])
        dcg = (hit_list / np.log2(1 + np.arange(1, 1 + len(hit_list)))).sum()
        idcg = (1 / np.log2(1 + np.arange(1, 1 + min(len(true), k)))).sum()
        total_ndcg += dcg / idcg
    
    return round(total_ndcg / len(y_true), 6)