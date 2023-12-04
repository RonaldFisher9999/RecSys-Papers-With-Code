import torch
from torch.utils.data import DataLoader
from LightGCN.model import LightGCN
import torch.nn as nn
from tqdm import tqdm


class LightGCNTrainer:
    def __init__(self,
                 num_epochs: int,
                 num_neg_samples: int,
                 lr: float,
                 batch_size: int,
                 lambda_reg: float,
                 device: str):
        self.num_epochs = num_epochs
        self.num_neg_samples = num_neg_samples
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.device = device
    
    def train(self,
              model: LightGCN,
              edge_index: torch.LongTensor):
        edge_index = edge_index.to(self.device)
        train_loader = DataLoader(range(edge_index.shape[1]), batch_size=self.batch_size, shuffle=True, drop_last=False)
        loss_fn = BPRLoss(self.lambda_reg)
        for epoch in range(1, self.num_epochs + 1):
            self.train_one_epoch()
            self.validate()
    
    def train_one_epoch(self,
                        model: LightGCN, 
                        edge_index: torch.LongTensor,
                        train_loader: DataLoader,
                        loss_fn: nn.Module,
                        ):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        neg_items = self.get_neg_items()
        total_loss = 0
        for index in tqdm(train_loader):
            user_index = edge_index[0, index]
            item_index = edge_index[1, index]
            pos_score = model.forward(edge_index, user_index, item_index)
            neg_score = 0
            for i in range(self.num_neg_samples):
                neg_score = neg_score + model.forward(edge_index, user_index, neg_items[i])
            
            if self.lambda_reg != 0:
                params = torch.cat([model.user_emb.weight, model.item_emb.weight], dim=0)
            else:
                params = None
            loss = self.loss_fn(pos_score, neg_score, params)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        
    
    def validate(self):
        pass
    
    def get_neg_items(self):
        pass
    
class BPRLoss(nn.Module):
    def __init__(self, lambda_reg: float = 0):
        super().__init__()
        self.lambda_reg = lambda_reg
        
    def forward(self, pos_score: torch.FloatTensor, neg_score: torch.FloatTensor, params: torch.FloatTensor = None):
        loss = nn.functional.logsigmoid(pos_score - neg_score).mean()
        reg_loss = 0
        if self.lambda_reg != 0 and params is not None:
            reg_loss = self.lambda_reg * params.norm(2).pow(2) / pos_score.shape[0]
            
        return -loss + reg_loss