import torch
import torch.nn.functional as F

def bpr_loss(pos_score: torch.Tensor,
             neg_score: torch.Tensor,
             **kwargs) -> torch.Tensor:
    return -F.logsigmoid(pos_score - neg_score).mean()

def bce_loss(pos_score: torch.Tensor,
             neg_score: torch.Tensor,
             **kwargs) -> torch.Tensor:
    beta = kwargs.get('beta', 1.0)
    score = torch.cat([pos_score, neg_score], dim=1)
    labels = torch.zeros_like(score).to(score.device)
    labels[:, 0] = 1
    pos_weight = torch.zeros(score.shape[1]).to(score.device)
    pos_weight[0] = beta
    return F.binary_cross_entropy_with_logits(score, labels, pos_weight=pos_weight, reduction='mean')