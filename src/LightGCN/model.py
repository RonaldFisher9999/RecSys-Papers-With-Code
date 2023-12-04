import torch
import torch.nn as nn
from torch_geometric.nn import LGConv
import numpy as np
from numpy.typing import NDArray


class LightGCN(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 num_layers: int,
                 emb_dim: int):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim)
        self.conv_layers = nn.ModuleList([LGConv() for _ in range(self.num_layers)])
        self.alpha = 1 / (self.num_layers + 1)
        
        self.user_final_emb = None
        self.item_final_emb = None
        
    def get_embedding(self, edge_index: torch.LongTensor) -> torch.Tensor:
        x_user = self.user_emb.weight
        x_item = self.item_emb.weight
        x = torch.cat([x_user, x_item], dim=0)

        out = x
        for i in range(self.num_layers):
            x = self.conv_layers[i].forward(x, edge_index)
            out = out + x
        out = out * self.alpha
        
        self.user_final_emb = out[:self.num_users].detach().clone()
        self.item_final_emb = out[self.num_users:].detach().clone()
        
        return out
    
    def forward(self,
                edge_index: torch.LongTensor,
                user_index: torch.LongTensor,
                item_index: torch.LongTensor) -> torch.Tensor:
        out = self.get_embedding(edge_index)
        out_user = out[user_index]
        out_item = out[item_index]
        score = (out_user * out_item).sum(dim=-1)
        
        return score
    
    def recommend(self, user_index: int, rated_items: NDArray[np.int64], k:int = 100) -> list[int]:
        score = self.user_final_emb[user_index] @ self.item_final_emb.T
        score[rated_items] = -float('inf')
        
        return torch.topk(score, k).indices.cpu().tolist()