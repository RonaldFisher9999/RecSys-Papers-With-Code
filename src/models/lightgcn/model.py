import torch
import torch.nn as nn
from torch_geometric.nn import LGConv
import numpy as np
from numpy.typing import NDArray

from models.common import BaseModel


class LightGCN(BaseModel):
    def __init__(self,
                 edge_index: torch.LongTensor,
                 num_users: int,
                 num_items: int,
                 num_layers: int,
                 emb_dim: int,
                 reg_2: float):
        super().__init__()
        self.edge_index = edge_index
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.reg_2 = reg_2
        
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim)
        self._reset_parameters()
        self.conv_layers = nn.ModuleList([LGConv() for _ in range(self.num_layers)])
        
        self.final_emb = None
    
    def _reset_parameters(self):
        for emb in self.modules():
            if isinstance(emb, nn.Embedding):
                torch.nn.init.xavier_uniform_(emb.weight)
    
    def _get_ego_embeddings(self) -> torch.Tensor:
        user_ego_emb = self.user_emb.weight
        item_ego_emb = self.item_emb.weight
        ego_emb = torch.cat([user_ego_emb, item_ego_emb], dim=0)
        
        return ego_emb
    
    def forward(self) -> torch.Tensor:
        emb = self._get_ego_embeddings()
        emb_list = [emb]

        for i in range(self.num_layers):
            emb = self.conv_layers[i].forward(emb, self.edge_index)
            emb_list.append(emb)
        
        final_emb = torch.stack(emb_list, dim=1).mean(dim=1)
        self.final_emb = final_emb.detach().clone()
        
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

    def recommend(self, user: int, rated_items: NDArray[np.int64], k: int=10) -> list[int]:
        user_final_emb = self.final_emb[:self.num_users, :]
        item_final_emb = self.final_emb[self.num_users:, :]
        score = user_final_emb[user] @ item_final_emb.T
        score[rated_items] = -float('inf')
        
        return torch.topk(score, k).indices.cpu().tolist()
    
    
def bpr_loss(pos_score: torch.Tensor,
             neg_score: torch.Tensor) -> torch.Tensor:
    return -nn.functional.logsigmoid(pos_score - neg_score).mean()