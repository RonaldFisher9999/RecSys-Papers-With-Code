import torch
import numpy as np
import random

from models.lightgcn.trainer import LightGCNTrainer
from models.common import BaseModelTrainer
from data.datamodel import BaseData
from config import Config, config_parser

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_training() -> Config:
    config = config_parser()
    for k, v in vars(config).items():
        print(f'{k}: {v}')
    set_seeds(config.seed)
    if config.device == 'cuda' and not torch.cuda.is_available():
        print('GPU not available. Use CPU instead.')
        config.device = 'cpu'
        
    return config
    
def build_model_trainer(config: Config, data: BaseData) -> BaseModelTrainer:
    print(f'Build "{config.model}" trainer.')
    if config.model == 'lightgcn':
        trainer = LightGCNTrainer(config, data)
    else:
        NotImplementedError('Other Models Are Not Implemented.')
        
    return trainer
