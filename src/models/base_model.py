import torch.nn as nn


class BaseModel(nn.Module):
    def forward(self):
        raise NotImplementedError
    
    def calc_loss(self):
        raise NotImplementedError
    
    def recommend(self):
        raise NotImplementedError
    