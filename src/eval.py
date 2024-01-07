import numpy as np


def recall_k(y_true: list[list[int]], y_pred: list[list[int]], k: int) -> float:
    total_recall = 0
    for true, pred in zip(y_true, y_pred):
        recall = len(set(true) & set(pred[:k])) / min(len(true), k)
        total_recall += recall
    
    return round(total_recall / len(y_true), 6)

def ndcg_k(y_true: list[list[int]], y_pred: list[list[int]], k: int) -> float:
    total_ndcg = 0
    for true, pred in zip(y_true, y_pred):
        true = set(true)
        hit_list = np.array([1 if x in true else 0 for x in pred[:k]])
        dcg = (hit_list / np.log2(1 + np.arange(1, 1 + len(hit_list)))).sum()
        idcg = (1 / np.log2(1 + np.arange(1, 1 + min(len(true), k)))).sum()
        total_ndcg += dcg / idcg
    
    return round(total_ndcg / len(y_true), 6)
