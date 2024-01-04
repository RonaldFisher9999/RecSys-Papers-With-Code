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
        total_neg_items[:, offset : offset + pos_items.size] = neg_items.reshape(-1,  pos_items.size)
        offset += pos_items.size
    total_neg_items += num_users
    
    return torch.tensor(total_neg_items, dtype=torch.int64)


class LightGCNTrainer(BaseModelTrainer):
    def __init__(self,
                 config: Config,
                 data: GraphModelData,
                 data_info: BaseDataInfo):
        super().__init__()
        self.num_epochs = config.num_epochs
        self.num_neg_samples = config.num_neg_samples
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.reg_2 = config.reg_2
        self.device = config.device
        self.num_users = data_info.num_users
        self.num_items = data_info.num_items
        self.num_layers = config.num_layers
        self.emb_dim = config.emb_dim
        self.best_model_path = os.path.join(config.checkpoint_dir, f'lightgcn_{config.dataset}.pt')
        self.best_score = 0.0
        self.model = self._build_model(data.edge_index).to(self.device)
    
    def train(self, data: GraphModelData):
        edge_index = data.edge_index
        train_loader = DataLoader(range(edge_index.shape[1] // 2), batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print(f'Start training for {self.num_epochs} epochs.')
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch}')
            total_neg_items = get_neg_items(data.rating_train, self.num_users, self.num_items, self.num_neg_samples)
            self._fit(edge_index, train_loader, total_neg_items, optimizer)
            recall, ndcg = self._validate('val', data.rating_train, data.rating_val)
            self._update_best_model(ndcg)
            
    def test(self, data: GraphModelData):
        self._load_best_model()
        self._validate('test', data.rating_train_val, data.rating_test)
    
    def _build_model(self, edge_index: torch.LongTensor):
        return LightGCN(edge_index.to(self.device),
                        self.num_users, self.num_items, self.num_layers, self.emb_dim,
                        self.reg_2)
        
    def _fit(self,
             edge_index: torch.LongTensor,
             train_loader: DataLoader,
             total_neg_items: torch.LongTensor,
             optimizer: torch.optim.Optimizer):
        print('Train model.')
        total_loss = 0
        for index in tqdm(train_loader):
            user_index = edge_index[0, index]
            pos_item_index = edge_index[1, index]
            neg_items = total_neg_items[:, index]
            loss = self.model.calc_loss(user_index, pos_item_index, neg_items)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f'Train Loss: {total_loss}')

    def _validate(self,
                  mode: str,
                  rating_train: pd.Series,
                  rating_valid: pd.Series,
                  k: int=20,) -> tuple[float, float]:
        if mode == 'val':
            print('Validate model.')
        if mode == 'test':
            print('Test model.')
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
        self.model = torch.load(self.best_model_path, map_location=self.device)
        