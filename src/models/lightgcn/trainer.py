import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from models.lightgcn.model import LightGCN
from models.common import BaseModelTrainer
from config import Config
from data.datamodel import BaseDataInfo, GraphModelData
from eval import ndcg_k, recall_k


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

class LightGCNTrainer(BaseModelTrainer):
    def __init__(self,
                 config: Config,
                 data_info: BaseDataInfo):
        super().__init__()
        self.num_epochs = config.num_epochs
        self.num_neg_samples = config.num_neg_samples
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.lambda_reg = config.lambda_reg
        self.device = config.device
        self.num_users = data_info.num_users
        self.num_items = data_info.num_items
        self.num_layers = config.num_layers
        self.emb_dim = config.emb_dim
        self.best_model_path = os.path.join(config.checkpoint_dir, f'lightgcn_{config.dataset}.pt')
        self.best_score = 0.0
        self.model = self._build_model().to(self.device)
    
    def train(self, data: GraphModelData):
        edge_index = data.edge_index.to(self.device)
        train_loader = DataLoader(range(edge_index.shape[1] // 2), batch_size=self.batch_size, shuffle=True, drop_last=False)
        loss_fn = BPRLoss(self.lambda_reg)
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch}')
            neg_items = get_neg_items(data.rating_train, self.num_items, self.num_neg_samples)
            self._fit(edge_index, train_loader, neg_items, loss_fn)
            recall, ndcg = self._validate(data.rating_train, data.rating_val)
            self._update_best_model(ndcg)
        
        return self._load_best_model()
    
    def _build_model(self):
        return LightGCN(self.num_users, self.num_items, self.num_layers, self.emb_dim)
        
    def _fit(self,
             edge_index: torch.LongTensor,
             train_loader: DataLoader,
             neg_items: torch.LongTensor,
             loss_fn: BPRLoss,
             ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        total_loss = 0
        for index in tqdm(train_loader):
            user_index = edge_index[0, index]
            item_index = edge_index[1, index]
            pos_score = self.model.forward(edge_index, user_index, item_index)
            neg_score = 0
            for i in range(neg_items.shape[0]):
                neg_item_index = neg_items[i, index]
                neg_score = neg_score + self.model.forward(edge_index, user_index, neg_item_index)
            neg_score = neg_score / neg_items.shape[0]
            if self.lambda_reg != 0:
                params = torch.cat([self.model.user_emb.weight, self.model.item_emb.weight], dim=0)
            else:
                params = None
            loss = loss_fn(pos_score, neg_score, params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f'Train Loss: {total_loss}')

    def _validate(self,
                 rating_train: pd.Series,
                 rating_valid: pd.Series,
                 k: int=10) -> tuple[float, float]:
        y_true = list()
        y_pred = list()
        for user in rating_train.index:
            rated_items = rating_train[user]
            pred = self.model.recommend(user, rated_items, k)
            true = rating_valid[user].tolist()
            y_true.append(true)
            y_pred.append(pred)
         
        recall = recall_k(y_true, y_pred, k)
        ndcg = ndcg_k(y_true, y_pred, k)
        print(f'Recall@{k}: {recall}')
        print(f'NDCG@{k}: {ndcg}')
        
        return recall, ndcg
    
    def _update_best_model(self, score: float):
        if self.best_score < score:
            print(f'Best Metric Updated {self.best_score} -> {score}')
            self.best_score = score
            torch.save(self.model, self.best_model_path)
            
    def _load_best_model(self):
        return torch.load(self.best_model_path, map_location=self.device)
