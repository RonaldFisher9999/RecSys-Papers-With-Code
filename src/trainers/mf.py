import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from models.mf import MF
from trainers.base_trainer import BaseModelTrainer
from config import Config
from data.datamodel import GraphModelData
from eval import ndcg_k, recall_k
from sampler import get_neg_items


class MFDataset(Dataset):
    def __init__(self,
                 u_i_index: np.ndarray,
                 rating: dict[int, np.ndarray],
                 num_items: int,
                 num_neg_samples: int):
        super().__init__()
        self.u_i_index = u_i_index
        self.num_items = num_items
        self.num_neg_samples = num_neg_samples
        self.neg_candi = self._create_candidate(rating)
        
    def _create_candidate(self, rating: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        print('Creating negative candidate items...')
        candi = dict()
        total_items = np.arange(self.num_items, dtype=np.int64)
        for user, items in rating.items():
            candi[user] = np.setdiff1d(total_items, items, assume_unique=True)
        return candi

    def __len__(self):
        return len(self.u_i_index)
    
    def __getitem__(self, idx: int):
        user_index = self.u_i_index[idx, 0]
        pos_item_index = self.u_i_index[idx, 1]
        neg_candi = self.neg_candi[user_index]
        neg_item_index = neg_candi[np.random.randint(0, neg_candi.size, size=self.num_neg_samples)]

        return user_index, pos_item_index, neg_item_index

class MFTrainer(BaseModelTrainer):
    def __init__(self,
                 config: Config,
                 data: GraphModelData):
        super().__init__()
        self.num_epochs = config.num_epochs
        self.num_neg_samples = config.num_neg_samples
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.device = config.device
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.model = self._build_model(self.num_users, self.num_items, config.emb_dim, config.loss)
        self.loader = self._build_loader(data.u_i_index, data.rating_train)
        self.best_model_path = os.path.join(config.checkpoint_dir, f'{config.model}_{config.dataset}.pt')
        self.best_score = 0.0
    
    def train(self, data: GraphModelData):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        print(f'Start training for {self.num_epochs} epochs.')
        for epoch in range(1, self.num_epochs + 1):
            print(f'Epoch {epoch}')
            self._fit(optimizer)
            recall, ndcg = self._validate('val', data.rating_train, data.rating_val)
            self._update_best_model(ndcg)
        print('Training done.')
        
    def test(self, data: GraphModelData):
        self._load_best_model()
        rating_train_val = self._merge_train_val(data.rating_train, data.rating_val)
        self._validate('test', rating_train_val, data.rating_test)

    def _build_model(self,
                     num_users: int,
                     num_items: int,
                     emb_dim: int,
                     loss: str):
        return MF(num_users, num_items, emb_dim, loss).to(self.device)
    
    def _build_loader(self,
                      u_i_index: np.ndarray,
                      rating: dict[int, np.ndarray]) -> DataLoader:
        dataset = MFDataset(u_i_index, rating, self.num_items, self.num_neg_samples)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def _fit(self, optimizer: torch.optim.Optimizer):
        print('Train model.')
        total_loss = 0
        self.model.train()
        for user_index, pos_item_index, neg_item_index in tqdm(self.loader):
            loss = self.model.calc_loss(user_index, pos_item_index, neg_item_index)
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
        self.model.eval()
        with torch.no_grad():
            for user, rated_items in rating_train.items():
                pred = self.model.recommend(user, rated_items, k)
                true = rating_valid[user]
                y_true.append(true.tolist())
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
        
    def _merge_train_val(self,
                         rating_train: dict[int, np.ndarray],
                         rating_val: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        rating_train_val = dict()
        for user in rating_train.keys():
            train_val = np.append(rating_train[user], rating_val[user])
            rating_train_val[user] = train_val
            
        return rating_train_val
        