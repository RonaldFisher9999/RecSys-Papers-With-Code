import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os

from models.bpr import BPR
from trainers.base_trainer import BaseModelTrainer
from config import Config
from data.datamodel import GraphModelData
from eval import ndcg_k, recall_k
from sampler import get_neg_items



class BPRTrainer(BaseModelTrainer):
    def __init__(self,
                 config: Config,
                 data: GraphModelData):
        super().__init__()
        self.num_epochs = config.num_epochs
        self.num_neg_samples = config.num_neg_samples
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.reg_2 = config.reg_2
        self.device = config.device
        self.emb_dim = config.emb_dim
        self.best_model_path = os.path.join(config.checkpoint_dir, f'{config.model}_{config.dataset}.pt')
        self.best_score = 0.0
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.model = self._build_model().to(self.device)
    
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
        print('Training done.')
        
    def test(self, data: GraphModelData):
        self._load_best_model()
        self._validate('test', data.rating_train_val, data.rating_test)
    
    def _build_model(self):
        return BPR(self.num_users, self.num_items, self.emb_dim, self.reg_2)
        
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
        for user in tqdm(rating_train.index):
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
        