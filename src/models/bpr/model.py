import torch
import torch.nn as nn
import numpy as np
from numpy.typing import NDArray

from models.common import BaseModel
from models.loss import bpr_loss


class BPR(BaseModel):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 emb_dim: int,
                 reg_2: float):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.reg_2 = reg_2
        
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim)
        self._reset_parameters()
        
    def _reset_parameters(self):
        for emb in self.modules():
            if isinstance(emb, nn.Embedding):
                torch.nn.init.xavier_uniform_(emb.weight)
                
    def forward(self) -> torch.Tensor:
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        final_emb = torch.cat([user_emb, item_emb], dim=0)
        
        return final_emb
    
    def calc_loss(self,
                  user_index: torch.LongTensor,
                  pos_item_index: torch.LongTensor,
                  neg_items: torch.LongTensor) -> torch.Tensor:
        '''
        user_index: (batch_size,)
        pos_item_index: (batch_size,)
        neg_items: (num_neg_samples, batch_size)
        '''
        final_emb = self.forward()

        user_emb = final_emb[user_index]
        pos_item_emb = final_emb[pos_item_index]
        pos_score = (user_emb * pos_item_emb).sum(dim=1)
        
        neg_score = 0
        for i in range(neg_items.shape[0]):
            neg_item_emb = final_emb[neg_items[i]]
            neg_score = neg_score + (user_emb * neg_item_emb).sum(dim=1)
        
        loss = bpr_loss(pos_score, neg_score)
        total_neg_item_index = neg_items.ravel()
        reg_loss = (user_emb.norm(2).pow(2)
                    + pos_item_emb.norm(2).pow(2)
                    + final_emb[total_neg_item_index].norm(2).pow(2)) * self.reg_2 / pos_score.shape[0]
        
        return loss + reg_loss
    
    def recommend(self, user: int, rated_items: NDArray[np.int64], k: int=20) -> list[int]:
        score = self.user_emb.weight[user] @ self.item_emb.weight.T
        score[rated_items] = -float('inf')
        
        return torch.topk(score, k).indices.cpu().tolist()
