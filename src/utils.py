import torch
import numpy as np
import random

from models.lightgcn.trainer import LightGCNTrainer
from config import Config

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_training(config: Config):
    for k, v in vars(config).items():
        print(k, v)
    set_seeds(config.seed)
    
def build_model_trainer(config: Config):
    if config.model == 'lightgcn':
        trainer = LightGCNTrainer(config)
    else:
        NotImplementedError('Other Models Are Not Implemented.')
        
    return trainer
