from args import parser
from model import LightGCN
from process import DataProcessor
from trainer import LightGCNTrainer
import random
import numpy as np
import torch


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main():
    config = parser.parse_args()
    for k, v in vars(config).items():
        print(k, v)
    set_seeds(config.seed)
    processor = DataProcessor(
        config.dataset,
        config.min_user_cnt,
        config.min_item_cnt,
        config.test_ratio,
        config.valid_ratio      
    )
    edge_index, train, valid, test, num_users, num_items = processor.process()
    # print(edge_index)
    # print(edge_index.shape)
    # print(train)
    # print(valid)
    # print(test)
    # print(n_users)
    # print(n_items)
    
    model = LightGCN(num_users, num_items, config.num_layers, config.emb_dim).to(config.device)
    print(model)
    
    trainer = LightGCNTrainer(config.num_epochs, num_users, num_items, config.num_neg_samples,
                              config.lr, config.batch_size, config.lambda_reg, config.device)
    trainer.train(model, edge_index, train, valid)
    
if __name__=='__main__':
    main()