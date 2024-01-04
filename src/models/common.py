import torch.nn as nn


class BaseModel(nn.Module):
    def forward(self):
        raise NotImplementedError
    
    def calc_loss(self):
        raise NotImplementedError
    
    def recommend(self):
        raise NotImplementedError
    

class BaseModelTrainer:
    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError
    
    def _build_model(self):
        raise NotImplementedError
    
    def _fit(self):
        raise NotImplementedError
    
    def _validate(self):
        raise NotImplementedError
    
    def _update_best_model(self):
        raise NotImplementedError
    
    def _load_best_model(self):
        raise NotImplementedError
    