import numpy as np
import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.loss import bce_loss, bpr_loss


class LightGCN(BaseModel):
    def __init__(
        self,
        adj_mat: torch.sparse.Tensor,
        num_users: int,
        num_items: int,
        num_layers: int,
        emb_dim: int,
        loss: str,
        device: str,
    ):
        super().__init__()
        self.norm_adj_mat = self._normalize_adj_mat(adj_mat).to(device)
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.loss = loss
        self.loss_fn = self._get_loss_function()

        self.user_emb = nn.Embedding(self.num_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim)
        self._reset_parameters()

        self.user_final_emb, self.item_final_emb = None, None

    def _normalize_adj_mat(self, adj_mat: torch.sparse.Tensor):
        degree = torch.sparse.sum(adj_mat, dim=1).to_dense()
        norm_degree = degree.pow(-0.5)
        norm_degree[norm_degree == float('inf')] = 0
        row, col = adj_mat.indices()[0], adj_mat.indices()[1]
        values = norm_degree[row] * norm_degree[col]
        norm_adj_mat = torch.sparse_coo_tensor(
            indices=adj_mat.indices(), values=values, size=adj_mat.size(), device=adj_mat.device
        )

        return norm_adj_mat.to_sparse_csr()

    def _get_loss_function(self):
        if self.loss == 'bpr':
            return bpr_loss
        elif self.loss == 'bce':
            return bce_loss
        else:
            raise NotImplementedError('Loss function other than "bpr" or "bce" is not implemented.')

    def _reset_parameters(self):
        for emb in self.modules():
            if isinstance(emb, nn.Embedding):
                torch.nn.init.xavier_uniform_(emb.weight)

    def _get_ego_embeddings(self) -> torch.Tensor:
        user_ego_emb = self.user_emb.weight
        item_ego_emb = self.item_emb.weight
        ego_emb = torch.cat([user_ego_emb, item_ego_emb], dim=0)

        return ego_emb

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self._get_ego_embeddings()
        emb_list = [emb]

        for _ in range(self.num_layers):
            emb = torch.sparse.mm(self.norm_adj_mat, emb)
            emb_list.append(emb)

        final_emb = torch.stack(emb_list, dim=1).mean(dim=1)
        user_final_emb = final_emb[: self.num_users, :]
        item_final_emb = final_emb[self.num_users :, :]

        return user_final_emb, item_final_emb

    def calc_loss(
        self,
        user_index: torch.LongTensor,
        pos_item_index: torch.LongTensor,
        neg_item_index: torch.LongTensor,
        **loss_kwargs
    ) -> torch.Tensor:
        '''
        user_index: (batch_size,)
        pos_item_index: (batch_size,)
        neg_item_index: (batch_size, num_neg_samples)
        '''
        user_final_emb, item_final_emb = self.forward()
        user_emb = user_final_emb[user_index]
        pos_item_emb = item_final_emb[pos_item_index]
        pos_score = (user_emb * pos_item_emb).sum(dim=-1).unsqueeze(1)

        neg_item_emb = item_final_emb[neg_item_index]
        neg_score = (user_emb.unsqueeze(1) * neg_item_emb).sum(dim=-1)

        return self.loss_fn(pos_score, neg_score, **loss_kwargs)

    def recommend(self, user: int, rated_items: np.ndarray, k: int, need_update: bool) -> list[int]:
        if need_update:
            self.user_final_emb, self.item_final_emb = self.forward()
        score = self.user_final_emb[user] @ self.item_final_emb.T
        score[rated_items] = -float('inf')

        return torch.topk(score, k).indices.cpu().tolist()
