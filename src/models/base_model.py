from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def calc_loss(self):
        raise NotImplementedError

    @abstractmethod
    def recommend(self):
        raise NotImplementedError
