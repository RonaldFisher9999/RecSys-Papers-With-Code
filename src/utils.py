import torch
import numpy as np
import random

from models.lightgcn.trainer import LightGCNTrainer
from models.common import BaseModelTrainer
from data.datamodel import BaseDataInfo, BaseData
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
    
def build_model_trainer(config: Config, data: BaseData, data_info: BaseDataInfo) -> BaseModelTrainer:
    print(f'Build "{config.model}" trainer.')
    if config.model == 'lightgcn':
        trainer = LightGCNTrainer(config, data, data_info)
    else:
        NotImplementedError('Other Models Are Not Implemented.')
        
    return trainer
