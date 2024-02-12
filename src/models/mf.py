import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray

from models.base_model import BaseModel
from models.loss import bpr_loss, bce_loss


class MF(BaseModel):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 emb_dim: int,
                 loss: str):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.loss = loss
        self.loss_fn = self._get_loss_function()
        
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for emb in self.modules():
            if isinstance(emb, nn.Embedding):
                torch.nn.init.xavier_uniform_(emb.weight)
    
    def _get_loss_function(self):
        if self.loss == 'bpr':
            return bpr_loss
        elif self.loss == 'bce':
            return bce_loss
        else:
            raise NotImplementedError('Loss function other than "bpr" or "bce" is not implemented.')
                
    def forward(self):
        pass
    
    def calc_loss(self,
                  user_index: torch.LongTensor,
                  pos_item_index: torch.LongTensor,
                  neg_item_index: torch.LongTensor,
                  **loss_kwargs) -> torch.Tensor:
        '''
        user_index: (batch_size,)
        pos_item_index: (batch_size,)
        neg_item_index: (batch_size, num_neg_samples)
        '''
        user_emb = self.user_emb.weight[user_index]
        pos_item_emb = self.item_emb.weight[pos_item_index]
        pos_score = (user_emb * pos_item_emb).sum(dim=-1).unsqueeze(1)
        
        neg_item_emb = self.item_emb.weight[neg_item_index]
        neg_score = (user_emb.unsqueeze(1) * neg_item_emb).sum(dim=-1)
        
        return self.loss_fn(pos_score, neg_score, **loss_kwargs)
    
    def recommend(self, user: int, rated_items: np.ndarray, k: int=20) -> list[int]:
        score = self.user_emb.weight[user] @ self.item_emb.weight.T
        score[rated_items] = -float('inf')
        
        return torch.topk(score, k).indices.cpu().tolist()
