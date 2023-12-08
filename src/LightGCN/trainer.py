import torch
from torch.utils.data import DataLoader
from model import LightGCN
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np


class BPRLoss(nn.Module):
    def __init__(self, lambda_reg: float = 0):
        super().__init__()
        self.lambda_reg = lambda_reg
        
    def forward(self,
                pos_score: torch.Tensor,
                neg_score: torch.Tensor,
                params: torch.Tensor = None) -> torch.Tensor:
        loss = nn.functional.logsigmoid(pos_score - neg_score).mean()
        reg_loss = 0
        if self.lambda_reg != 0 and params is not None:
            reg_loss = self.lambda_reg * params.norm(2).pow(2) / pos_score.shape[0]
            
        return -loss + reg_loss
    
 
def get_neg_items(rating: pd.Series, n_items: int, num_neg_samples: int):
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
    
    return round(total_recall / len(y_true), 3)

def ndcg_k(y_true: list[list[int]], y_pred: list[list[int]],  k: int):
    total_ndcg = 0
    for true, pred in zip(y_true, y_pred):
        true = set(true)
        hit_list = np.array([1 if x in true else 0 for x in pred[:k]])
        dcg = (hit_list / np.log2(1 + np.arange(1, 1 + len(hit_list)))).sum()
        idcg = (1 / np.log2(1 + np.arange(1, 1 + min(len(true), k)))).sum()
        total_ndcg += dcg / idcg
    
    return round(total_ndcg / len(y_true), 3)

   
class LightGCNTrainer:
    def __init__(self,
                 num_epochs: int,
                 num_users: int,
                 num_items: int,
                 num_neg_samples: int,
                 lr: float,
                 batch_size: int,
                 lambda_reg: float,
                 device: str):
        self.num_epochs = num_epochs
        self.num_users = num_users
        self.num_items = num_items
        self.num_neg_samples = num_neg_samples
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.device = device
    
    def train(self,
              model: LightGCN,
              edge_index: torch.LongTensor,
              rating_train: pd.Series,
              rating_valid: pd.Series):
        edge_index = edge_index.to(self.device)
        train_loader = DataLoader(range(edge_index.shape[1] // 2), batch_size=self.batch_size, shuffle=True, drop_last=False)
        loss_fn = BPRLoss(self.lambda_reg)
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch}')
            neg_items = get_neg_items(rating_train, self.num_items, self.num_neg_samples)
            self.train_one_epoch(model, edge_index, train_loader, neg_items, loss_fn)
            self.validate(model, rating_train, rating_valid)
    
    def train_one_epoch(self,
                        model: LightGCN, 
                        edge_index: torch.LongTensor,
                        train_loader: DataLoader,
                        neg_items: torch.LongTensor,
                        loss_fn: BPRLoss,
                        ):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        total_loss = 0
        for index in tqdm(train_loader):
            user_index = edge_index[0, index]
            item_index = edge_index[1, index]
            pos_score = model.forward(edge_index, user_index, item_index)
            neg_score = 0
            for i in range(neg_items.shape[0]):
                neg_item_index = neg_items[i, index]
                neg_score = neg_score + model.forward(edge_index, user_index, neg_item_index)
            
            if self.lambda_reg != 0:
                params = torch.cat([model.user_emb.weight, model.item_emb.weight], dim=0)
            else:
                params = None
            loss = loss_fn(pos_score, neg_score, params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f'Train Loss: {total_loss}')

    def validate(self,
                 model: LightGCN,
                 rating_train: pd.Series,
                 rating_valid: pd.Series,
                 k: int=20):
        y_true = list()
        y_pred = list()
        for user in rating_train.index:
            rated_items = rating_train[user]
            pred = model.recommend(user, rated_items)
            true = rating_valid[user].tolist()
            y_true.append(true)
            y_pred.append(pred)
        
        print(f'Recall@{k}: {recall_k(y_true, y_pred, k)}')
        print(f'NDCG@{k}: {ndcg_k(y_true, y_pred, k)}')
        

    
