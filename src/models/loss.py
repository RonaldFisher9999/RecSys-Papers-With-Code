import torch
import torch.nn as nn

def bpr_loss(pos_score: torch.Tensor,
             neg_score: torch.Tensor) -> torch.Tensor:
    return -nn.functional.logsigmoid(pos_score - neg_score).mean()