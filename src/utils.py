import torch
import numpy as np
import random
import os

from trainers import BaseModelTrainer, MFTrainer, LightGCNTrainer
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
    
    if config.device == 'cuda' and not torch.cuda.is_available():
        print('GPU not available. Use CPU instead.')
        config.device = 'cpu'
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    set_seeds(config.seed)
    
    for k, v in vars(config).items():
        print(f'{k}: {v}')
        
    return config
    
def build_model_trainer(config: Config, data: BaseData) -> BaseModelTrainer:
    print(f'Build "{config.model}" trainer.')
    if config.model == 'lightgcn':
        trainer = LightGCNTrainer(config, data)
    elif config.model == 'mf':
        trainer = MFTrainer(config, data)
    else:
        NotImplementedError('Other models are not implemented.')
   
    return trainer
