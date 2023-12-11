import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import os

from models.lightgcn.model import LightGCN
from utils import get_neg_items, recall_k, ndcg_k


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

   
class LightGCNTrainer:
    def __init__(self,
                 num_epochs: int,
                 num_users: int,
                 num_items: int,
                 num_neg_samples: int,
                 lr: float,
                 batch_size: int,
                 lambda_reg: float,
                 device: str,
                 checkpoint_dir: str,
                 dataset: str):
        self.num_epochs = num_epochs
        self.num_users = num_users
        self.num_items = num_items
        self.num_neg_samples = num_neg_samples
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.device = device
        self.best_model_path = os.path.join(checkpoint_dir, f'lightgcn_{dataset}.pt')
        self.best_score = 0.0
    
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
            recall, ndcg = self.validate(model, rating_train, rating_valid)
            self.update_model(model, ndcg)
    
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
            neg_score = neg_score / neg_items.shape[0]
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
                 k: int=10) -> tuple[float, float]:
        y_true = list()
        y_pred = list()
        for user in rating_train.index:
            rated_items = rating_train[user]
            pred = model.recommend(user, rated_items, k)
            true = rating_valid[user].tolist()
            y_true.append(true)
            y_pred.append(pred)
        
        recall = recall_k(y_true, y_pred, k)
        ndcg = ndcg_k(y_true, y_pred, k)
        print(f'Recall@{k}: {recall}')
        print(f'NDCG@{k}: {ndcg}')
        
        return recall, ndcg
    
    def update_model(self, model: LightGCN, score: float):
        if self.best_score < score:
            print(f'Best Metric Updated {self.best_score} -> {score}')
            self.best_score = score
            torch.save(model, self.best_model_path)

    
